from lcd_clipr.definitions import SCRATCH_STORE_DIR

import math
import anndata as ad
import numpy as onp
from scipy import sparse
import torch
import tempfile
# torch.use_deterministic_algorithms(True)
# causes error: RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility


from lcd_clipr.core import normalizer
from lcd_clipr.baselines.cpa_repo import CPA


model_params = {
    "n_latent": 32,
    "recon_loss": "nb",
    "doser_type": "linear",
    "n_hidden_encoder": 256,
    "n_layers_encoder": 2,
    "n_hidden_decoder": 256,
    "n_layers_decoder": 2,
    "use_batch_norm_encoder": True,
    "use_layer_norm_encoder": False,
    "use_batch_norm_decoder": False,
    "use_layer_norm_decoder": False,
    "dropout_rate_encoder": 0.2,
    "dropout_rate_decoder": 0.0,
    "variational": False,
    "seed": 0,
}

trainer_params = {
    "n_epochs_kl_warmup": None,
    "n_epochs_adv_warmup": 50,
    "n_epochs_mixup_warmup": 10,
    "n_epochs_pretrain_ae": 10,
    "mixup_alpha": 0.1,
    "lr": 0.0001,
    "wd": 1e-06,
    "adv_steps": 3,
    "reg_adv": 10.0,
    "pen_adv": 20.0,
    "adv_lr": 0.0001,
    "adv_wd": 1e-06,
    "n_layers_adv": 2,
    "n_hidden_adv": 128,
    "use_batch_norm_adv": True,
    "use_layer_norm_adv": False,
    "dropout_rate_adv": 0.3,
    "step_size_lr": 25,
    "do_clip_grad": False,
    "adv_loss": "cce",
    "gradient_clip_value": 5.0,
}


def intervention_to_guide_id(intv, var_names):
    targets = onp.where(intv == 1)[0]
    if len(targets) == 0:
        return "ctrl"
    elif len(targets) == 1:
        return f"{var_names[targets[0]]}"
    elif len(targets) == 2:
        return f"{var_names[targets[0]]}+{var_names[targets[1]]}"
    else:
        raise ValueError("More than 2 targets are not supported")


def run_cpa(seed, config, train_data_noisy, train_intv, test_intv, var_names, *, n):
    torch.manual_seed(seed)
    rng = onp.random.default_rng(seed)
    assert len(train_data_noisy) == len(train_intv)

    model_params.update({
        "n_latent": config["n_latent"],
        "n_layers_encoder": config["layers"],
        "n_layers_decoder": config["layers"],
        "n_hidden_encoder": config["hidden"],
        "n_hidden_decoder": config["hidden"],
        "dropout_rate_encoder": config["dropout"],
        "dropout_rate_decoder": config["dropout"],
    })
    trainer_params.update({
        "n_layers_adv": config["layers"],
        "n_hidden_adv": config["hidden"],
        "reg_adv": config["reg_adv"], # strength of adversarial loss
        "wd": config["wd"],
        "adv_wd": config["wd"],
        "doser_wd": config["wd"],
        "dropout_rate_adv": config["dropout"],
    })

    ctrl = train_data_noisy[0]

    """
    Setup data in expected format
    """
    # train and valid
    all_data = []
    all_split = []
    all_guide_ids = []
    train_guide_ids = []
    for data, intv in zip(train_data_noisy, train_intv):
        guide = intervention_to_guide_id(intv, var_names)
        train_guide_ids.append(guide)

        all_guide_ids.append([guide] * data.shape[0])
        all_data.append(data)

    # determine which doubles are used for validation
    available_doubles = [g for g in train_guide_ids if "+" in g]
    validation_doubles = rng.permutation(available_doubles)[:math.ceil(len(available_doubles) * 0.2)].tolist()
    for guide, guides in zip(train_guide_ids, all_guide_ids):
        if guide in validation_doubles:
            all_split.append(["valid"] * len(guides))
        else:
            all_split.append(["train"] * len(guides))

    # test: random samples of control samples
    test_guide_ids = []
    for intv in test_intv:
        guide = intervention_to_guide_id(intv, var_names)
        test_guide_ids.append(guide)

        # control samples from which we later generate the counterfactual
        all_guide_ids.append([guide] * n)
        all_data.append(rng.choice(ctrl, size=n, replace=True, axis=0))
        all_split.append(["test"] * n)

    adata = ad.AnnData(sparse.csr_matrix(onp.matrix(onp.concatenate(all_data, axis=0))))
    adata.obs["condition"] = onp.concatenate(all_guide_ids)
    adata.var["gene_name"] = var_names
    adata.obs["split"] = onp.concatenate(all_split)

    for split in ["train", "valid", "test"]:
        print(split)
        print(adata[adata.obs['split'] == split].obs['condition'].value_counts())


    # init CPA
    CPA.setup_anndata(
        adata,
        perturbation_key='condition',
        control_group='ctrl',
        is_count_data=True,
        max_comb_len=2,
    )

    model = CPA(
        adata=adata,
        split_key='split',
        train_split='train',
        valid_split='valid',
        test_split='test',
        **model_params,
    )

    # training (use $SCRATCH to avoid filling /tmp or current directory)
    with tempfile.TemporaryDirectory(dir=SCRATCH_STORE_DIR) as scratch_dir:
        model.train(
            max_epochs=1000,
            use_gpu=True,
            plan_kwargs=trainer_params,
            save_path=scratch_dir,
            enable_progress_bar=False,
            enable_checkpointing=False,
            default_root_dir=scratch_dir,  # Prevent Lightning logs in current directory
        )

    # predictions: set X to a sample ctrl data and then call .predict
    model.predict(adata)
    adata_test = adata[adata.obs["split"] == "test"]

    samples = []
    for test_guide in test_guide_ids:
        pred_counts = adata_test[adata_test.obs['condition'] == test_guide].copy().obsm['CPA_pred']
        samples.append(pred_counts)

    norm_samples = normalizer(samples)

    return norm_samples
