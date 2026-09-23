from collections import defaultdict
import time

import numpy as onp
import torch
torch.use_deterministic_algorithms(True)

import torch.nn as nn
from lcd_clipr.utils.opt import update_ave, retrieve_ave, at_t


class PEPER(nn.Module):
    def __init__(self, train_data, train_intv, *, hidden_size, encoder_layers, decoder_layers):
        super().__init__()

        # init SALT component
        self.__init_salt_component(train_data, train_intv)

        # init MLP component
        nonlinearity = nn.ReLU()
        self.d = train_intv.shape[1]
        self.hidden_size = hidden_size

        f1 = []
        for i in range(encoder_layers):
            f1.append(nn.Linear(hidden_size if i else self.d, hidden_size))
            f1.append(nonlinearity)
        self.f1 = nn.Sequential(*f1)

        f2 = []
        for _ in range(decoder_layers):
            f2.append(nn.Linear(hidden_size, hidden_size))
            f2.append(nonlinearity)
        f2.append(nn.Linear(hidden_size, self.d))
        self.f2 = nn.Sequential(*f2)


    def __init_salt_component(self, train_data, train_intv):
        assert len(train_data) == len(train_intv)
        self.ctrl = train_data[0]
        d = self.ctrl.shape[-1]

        # compute means of single-gene perturbations for each training environment
        self.ctrl_mean = self.ctrl.mean(0)
        perturbed_mean = onp.concatenate(train_data[1:]).mean(0)

        means = defaultdict(list)
        for data, intv in zip(train_data[1:], train_intv[1:]):
            if intv.sum() == 1:
                tar = onp.where(intv == 1)[0][0]
                means[tar].append(data.mean(0))

        mu = onp.array([(onp.stack(means[j]).mean(0) if j in means else perturbed_mean) for j in range(d)])
        mu -= self.ctrl_mean
        self.mu = torch.tensor(mu)


    def forward(self, intv):
        pred = torch.tensor(self.ctrl_mean)
        tars = onp.where(intv == 1)[0]

        # salt component
        for tar in tars:
             pred += self.mu[tar]

        # peper component
        peper_embedding = torch.zeros(self.hidden_size)
        for tar in tars:
            peper_embedding += self.f1(self.mu[tar])
        pred += self.f2(peper_embedding)

        return pred


    def predict_all(self, test_intv):
        return torch.stack([self.forward(intv) for intv in test_intv]).detach().numpy()


    def sample_all(self, rng, test_intv, n):
        # predict shifts
        means = self.predict_all(test_intv)

        # estimate distribution of each test environment by perturbing control samples
        samples = []
        for mean in means:
            ctrl_sample = rng.permutation(self.ctrl)[:n]
            samples.append(ctrl_sample + mean - self.ctrl_mean)

        samples = onp.stack(samples)
        return samples


def run_peper(seed, config, train_data, train_intv, test_intv, *, n):
    torch.manual_seed(seed)
    assert len(train_data) == len(train_intv)

    # init model
    model = PEPER(train_data, train_intv,
                  hidden_size=config["hidden_size"],
                  encoder_layers=config["layers"],
                  decoder_layers=config["layers"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    def cmd_loss(pred, data):
        if config["loss"] == "cmd":
            return ((pred - data) ** 2).sum().sqrt()
        elif config["loss"] == "mse":
            return ((pred - data) ** 2).sum()
        else:
            raise ValueError(f"Unknown PEPER loss: {loss}")

    # training loop
    t_loop = time.time()
    logs = defaultdict(float)
    train_data = [torch.tensor(data) for data in train_data]
    log_every = config["steps"] / 20

    for t in range(config["steps"]):
        # make batch
        env = torch.randint(0, len(train_data), (1,)).item()
        idx = torch.randperm(len(train_data[env]))[:config["batch_size"]]
        intv = train_intv[env]
        batch = train_data[env][idx]

        # compute loss and backprop
        optimizer.zero_grad()
        loss = cmd_loss(model(intv), batch)
        loss.backward()

        # update step
        optimizer.step()

        # log
        logs = update_ave(logs, {
            "loss": loss.item(),
        })
        if at_t(t, log_every):
            t_elapsed = time.time() - t_loop
            t_loop = time.time()

            ave_logs = retrieve_ave(logs)
            logs = defaultdict(float)

            print_str = f"t: {t: >5d}  loss: {ave_logs['loss']: >12.6f}"
            print_str += f"\tmin: {(config['steps'] - t) * t_elapsed / log_every / 60.0: >4.1f} "
            print_str += f"\tsec/1k steps: {1000 * t_elapsed / log_every: >4.2f} "
            print(print_str, flush=True)


    # make predictions
    with torch.no_grad():
        nprng = onp.random.default_rng(seed)
        samples = model.sample_all(nprng, test_intv, n)

    return samples