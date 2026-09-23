import math
import numpy as onp
import jax.numpy as jnp

def update_ave(ave_d, d):
    # online mean and variance with Welfords algorithm
    for k, v in d.items():
        # handle cases where a metric is not measured (nan on purpose)
        if onp.isnan(v) and "dsm_loss__" in k:
            continue

        ave_d[("__ctr__", k)] += 1
        delta = v - ave_d[("__mean__", k)]
        ave_d[("__mean__", k)] += delta / ave_d[("__ctr__", k)]
        delta2 = v - ave_d[("__mean__", k)]
        ave_d[("__welford_m2__", k)] += delta * delta2
    return ave_d


def retrieve_ave(ave_d, mean_only=True):
    out = dict(mean={}, std={})
    for k, v in ave_d.items():
        assert isinstance(k, tuple)
        # check if `k` is a ctr element
        if k[0] == "__ctr__":
            continue
        # process value `v`
        try:
            v_val = v.item()
        # array case
        except TypeError:
            v_val = onp.array(v)
        # not an array
        except AttributeError:
            v_val = v
        assert ("__ctr__", k[1]) in ave_d.keys()
        if k[0] == "__mean__":
            out["mean"][k[1]] = v_val
        else:
            if ave_d[("__ctr__", k[1])] == 1:
                out["std"][k[1]] = 0.0
            else:
                # sample std (div. by n-1)
                # out["std"][k[1]] = math.sqrt((v_val + 1e-18) / (ave_d[("__ctr__", k[1])] - 1))

                # std  (div. by n)
                out["std"][k[1]] = math.sqrt((v_val + 1e-18) / (ave_d[("__ctr__", k[1])]))

    if mean_only:
        return out["mean"]
    else:
        return out


def at_t(t, freq, exclude_zero=True) -> bool:
    if t == 0 and exclude_zero:
        return False
    elif freq is None or freq == 0:
        return False
    if isinstance(freq, (list, tuple)):
        return (t + 1) in freq
    else:
        return (t + 1) % math.floor(freq) == 0


def update_theta_config(param, step, interval_tree):
    if "__config__" in param:
        config = param["__config__"]
        for k, v in interval_tree.items():
            for mode, intervals in v.items():
                if intervals == "x":
                    on = jnp.array(0.0)
                else:
                    intervals = intervals.split(",")
                    intervals = [lohi.split("-") for lohi in intervals]
                    intervals_on = jnp.array([
                        jnp.logical_and(
                            True if lo == "" else int(lo) < step,
                            True if hi == "" else step < int(hi)
                        ) for lo, hi in intervals
                    ])
                    on = intervals_on.any().astype(jnp.float32)
                config[mode][k] = config[mode][k].at[...].set(on)
    return param
