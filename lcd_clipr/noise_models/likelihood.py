from functools import partial

import jax
import numpy as onp
import scipy as sp
from jax import numpy as jnp, random, vmap, scipy as jsp
from jax._src.scipy.special import gammaln


def compute_unit_scaling(x, min_scaling=1.0):
    numpy = onp if type(x) == onp.ndarray else jnp
    assert x.ndim == 2
    sample_mean = x.mean(0)
    is_nonzero = ~numpy.isclose(sample_mean, 0.0)
    sample_mean_nonzero = numpy.where(is_nonzero, sample_mean, 1.0)
    sample_var = x.var(0, ddof=1)
    zero_inflated_poisson_var_mle = numpy.where(is_nonzero, ((sample_var + (sample_mean ** 2)) / sample_mean_nonzero) - 1, 0.0)
    if min_scaling < 0:
        unit_scaling = numpy.maximum(numpy.sqrt(zero_inflated_poisson_var_mle), min_scaling)
    else:
        unit_scaling = numpy.maximum(zero_inflated_poisson_var_mle, min_scaling)
    return unit_scaling


def pull_unit_scaling_into_state_space(x, unit_scaling, softplus_thres=16.0):
    """
    Implements the function f(x, c) = log((1 + exp(x))^c - 1) in a numerically stable way.
    Uses the following facts
        For x to inf, softplus becomes the identity
        For x to -inf, (1+exp(x))^c is approximately 1 + c * exp(x) (binomial approximation)
    f(x, c) satisfies the property that
        c * log(1 + exp(x)) = log(1 + exp(f(x, c))
    """
    numpy = onp if type(x) == onp.ndarray else jnp
    scipy = sp if type(x) == onp.ndarray else jsp
    assert softplus_thres > 0.0
    x_above_thres = x > softplus_thres
    x_below_thres = x < - softplus_thres
    inner = numpy.where(
        x_above_thres,
        unit_scaling * x,
        unit_scaling * numpy.logaddexp(0, x),
    )
    inner_above_thres = inner > softplus_thres
    stacked = onp.stack([inner, onp.zeros_like(inner)], axis=-1)
    f_unstable = scipy.special.logsumexp(stacked, b=numpy.array([1, -1]), axis=-1)
    f = numpy.where(
        inner_above_thres,
        inner,
        f_unstable,
    )
    f = numpy.where(
        x_below_thres,
        x + numpy.log(unit_scaling), # binomial approximation
        f,
    )
    return f


def link_state(s, temp=1.0, shift=0.0):
    numpy = onp if type(s) == onp.ndarray else jnp
    t = abs(temp)
    s += shift
    if temp > 0:
        s = numpy.logaddexp(0, s / t)
    else:
        s = numpy.exp(s / t)
    return s


def warp(s, soft=True, scaling=1.0, temp=0.0):
    if temp == 0.0 and scaling == 1.0:
        return s

    assert s.ndim >= 2
    numpy = onp if type(s) == onp.ndarray else jnp
    if soft:
        clip = lambda z: numpy.logaddexp(0, z / temp) * temp
    else:
        clip = lambda z: numpy.maximum(0, z)

    m = s.mean(-2, keepdims=True)
    if temp == 0.0:
        warp_factor = 1.0
    else:
        v = s.var(-2, keepdims=True)
        v_warped = clip(v - m)
        warp_factor = numpy.sqrt(v_warped / (m + v_warped))

    warp_factor *= scaling
    return (s - m) * warp_factor + m


def state_to_warped_poisson_rate(*, s, unit_scaling, link_temp, link_shift, warp_temp, warp_scaling):
    # link
    poisson_rate = link_state(s, temp=link_temp, shift=link_shift)

    # unit scaling
    poisson_rate *= unit_scaling

    # warping
    warped_poisson_rate = warp(poisson_rate, scaling=warp_scaling, temp=warp_temp)
    return warped_poisson_rate


@partial(vmap, in_axes=(0, 0), out_axes=0)
def poisson_log_likelihood(x, rate):
    assert rate.ndim == x.ndim == 0

    log_rate = jnp.log(rate)

    logp = (
        jnp.multiply(x, log_rate)
        - rate
        - gammaln(x + 1)
    )
    return logp


@partial(vmap, in_axes=(0, 0, 0), out_axes=0)
def zipoisson_log_likelihood(x, rate, logit_pi):
    assert rate.ndim == x.ndim == logit_pi.ndim == 0

    log_rate = jnp.log(rate)
    log_pi = - jnp.logaddexp(0, -logit_pi)
    log_1_minus_pi = - jnp.logaddexp(0, logit_pi)

    log_poisson_term = (
        jnp.multiply(x, log_rate)
        - rate
        - gammaln(x + 1)
    )

    common_term = log_1_minus_pi + log_poisson_term

    # compute cases and then combine using mask
    logp_zero = jnp.logaddexp(log_pi, common_term)
    logp_nonzero = common_term

    x_is_zero = jnp.isclose(x, 0)
    logp = x_is_zero * logp_zero + (1 - x_is_zero) * logp_nonzero
    return logp


@partial(vmap, in_axes=(0, None, None), out_axes=0) # over batch of datapoints
@partial(vmap, in_axes=(None, 0, None), out_axes=0) # over mc samples of warped_rate
def log_likelihood_given_distribution_params(x, warped_rate, logit_pi):
    assert x.ndim == 1
    assert x.shape == warped_rate.shape
    assert logit_pi is None or logit_pi.shape == warped_rate.shape
    zero_inflation = logit_pi is not None

    # stability of gamma_ln
    warped_rate += 1e-20

    if zero_inflation:
        loglik_per_gene = zipoisson_log_likelihood(x, warped_rate, logit_pi)
    else:
        loglik_per_gene = poisson_log_likelihood(x, warped_rate)

    return loglik_per_gene.sum(0)


def zipoisson_generative_model(key, s, param, *, unit_scaling, link_temp, link_shift, warp_scaling, warp_temp):
    # apply link function and then warp distribution to obtain poisson rate
    warped_poisson_rate = state_to_warped_poisson_rate(
        s=s,
        unit_scaling=unit_scaling,
        link_temp=link_temp,
        link_shift=link_shift,
        warp_temp=warp_temp,
        warp_scaling=warp_scaling,
    )

    # sample from poisson
    key, subk = random.split(key)
    x = random.poisson(subk, warped_poisson_rate)
    assert x.shape == s.shape

    # zero-inflation
    if "logit_pi" in param:
        pi = jax.nn.sigmoid(param["logit_pi"])
        key, subk = random.split(key)
        h = random.bernoulli(subk, p=pi, shape=x.shape).astype(jnp.int32)
        x *= (1 - h)

    return x
