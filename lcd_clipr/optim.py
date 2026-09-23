import optax
import numpy as onp

from lcd_clipr.utils.version_control import str2bool


def _init_scheduler(*, config, steps, minimum_value=0.0):
    if not type(config) == str:
        config = f"const__{config}"

    configs = config.split("__")

    if configs[0] == "const":
        assert len(configs) == 2
        sched = optax.constant_schedule(
            float(configs[1]),
        )

    elif configs[0] == "lin":
        assert len(configs) == 2
        sched = optax.linear_schedule(
            init_value=float(configs[1]),
            end_value=minimum_value,
            transition_steps=steps,
        )

    elif configs[0] == "exp":
        assert len(configs) >= 2
        init_value = float(configs[1])
        percent_close_to_end_value = float(configs[2]) if len(configs) > 2 else 0.01
        decay_rate = 0.5 # arbitrary, since decay determined by `percent_close_to_end_value`
        transition_steps = onp.ceil(
            steps * onp.log(decay_rate)
            / (onp.log(percent_close_to_end_value) + onp.log(init_value - minimum_value) - onp.log(init_value))
        )
        sched = optax.exponential_decay(
            init_value=init_value,
            end_value=minimum_value,
            transition_steps=transition_steps,
            decay_rate=decay_rate,
        )

    elif configs[0] == "cos":
        assert len(configs) >= 2
        init_value = float(configs[1])
        exponent = float(configs[2]) if len(configs) > 2 else 1.0
        minimum_value = float(configs[3]) if len(configs) > 3 else minimum_value
        alpha = minimum_value / init_value
        sched = optax.cosine_decay_schedule(
            init_value=init_value,
            decay_steps=steps,
            exponent=exponent,
            alpha=alpha,
        )

    elif configs[0] == "warmup_cos":
        assert len(configs) >= 3
        warmup_steps = int(configs[2])
        sched = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=float(configs[1]),
            warmup_steps=warmup_steps if warmup_steps < steps else 0,
            decay_steps=steps,
            exponent=float(configs[3]) if len(configs) > 3 else 1.0,
            end_value=minimum_value,
        )

    else:
        raise ValueError(f"Unknown learning rate schedule: {configs}")

    return sched


def init_scheduler(*, config, steps, minimum_value=0.0):
    if type(config) == tuple or type(config) == list:
        schedulers = []
        for conf in config:
            schedulers.append(_init_scheduler(config=conf, steps=steps, minimum_value=minimum_value))
        return schedulers
    else:
        return _init_scheduler(config=config, steps=steps, minimum_value=minimum_value)


def init_optimizer(*, optimizer, learning_rate, steps, grad_clip=None, weight_decay=0.0):

    optimizer_modules = []
    if grad_clip is not None and not grad_clip == 0.0:
        if grad_clip > 0:
            optimizer_modules.append(optax.clip_by_global_norm(abs(grad_clip)))
        else:
            optimizer_modules.append(optax.clip_by_block_rms(abs(grad_clip)))
    
    learning_rate_schedule = init_scheduler(config=learning_rate, steps=steps)

    configs = optimizer.split("__")

    if configs[0] == "sgd":
        assert len(configs) >= 1
        kwargs = dict()
        if weight_decay:
            optimizer_modules.append(optax.add_decayed_weights(weight_decay, True))
        if len(configs) > 1 and float(configs[1]):
            kwargs["momentum"] = float(configs[1])
        if len(configs) > 2:
            kwargs["nesterov"] = str2bool(configs[2])
        optimizer_modules.append(optax.sgd(learning_rate_schedule, **kwargs))

    elif configs[0] == "adam":
        assert len(configs) >= 1
        kwargs = dict(weight_decay=weight_decay)
        if len(configs) > 1:
            kwargs["b1"] = float(configs[1])
        if len(configs) > 2:
            kwargs["b2"] = float(configs[2])
        optimizer_modules.append(optax.adamw(learning_rate_schedule, **kwargs))

    elif configs[0] == "lamb":
        assert len(configs) >= 1
        kwargs = dict(weight_decay=weight_decay)
        if len(configs) > 1:
            kwargs["b1"] = float(configs[1])
        if len(configs) > 2:
            kwargs["b2"] = float(configs[2])
        optimizer_modules.append(optax.lamb(learning_rate_schedule, **kwargs))

    elif configs[0] == "adagrad":
        assert len(configs) >= 1
        optimizer_modules.append(optax.adagrad(learning_rate_schedule))

    else:
        raise ValueError(f"Unknown optimizer {configs[0]}")

    opt = optax.chain(*optimizer_modules)
    return opt, learning_rate_schedule

