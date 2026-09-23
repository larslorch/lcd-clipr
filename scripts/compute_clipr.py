import argparse
from pathlib import Path
import numpy as onp
from joblib import dump
from lcd_clipr.experiment.plot_config import *

import matplotlib.pyplot as plt
plt.rcParams.update(MATPLOTLIB_RCPARAMS)

from lcd_clipr.experiment.analyze_causal import causal_mats, empirical_differential_expressions
from lcd_clipr.utils.parse import load_json_zip, load_data
from lcd_clipr.models.main import Model
from lcd_clipr.utils.cache import init_cache
from lcd_clipr.metrics import compute_differential_expression_sign
from lcd_clipr.definitions import PROJECT_DIR, SUBDIR_RESULTS, SUBDIR_PLOTS, SUBDIR_CLIPR_DUMP


# memory = init_cache(ignore=True)
memory = init_cache(ignore=False)


@memory.cache
def load_json_zip_cached(path):
    return load_json_zip(path)


@memory.cache
def load_data_cached(path):
    return load_data(path)


@memory.cache
def compute_differential_expression_sign_cached(pred, ctrl, single_gene_only, return_details, fdr_threshold, targets):
    return compute_differential_expression_sign(
        pred,
        ctrl,
        single_gene_only=single_gene_only,
        return_details=return_details,
        fdr_threshold=fdr_threshold,
        targets=targets,
    )


def compute_and_save_clipr(
        pred,
        base_save_path,
        model,
        param,
        train_data_normalized,
        train_intv,
        *,
        tikhonov,
        test_data_normalized=None,
        test_intv=None,
):

    var_names = pred["data_seeding"]["var_names"]
    gene_to_ensembl = pred["data_seeding"]["gene_to_ensembl"]

    # standardization info
    standardization_aux = pred["standardization_aux"]
    scale_standardization = pred["config"]["scale_standardization"]
    assert scale_standardization == "all-std"

    shift = standardization_aux["ctrl_mean"]
    scale = standardization_aux["all_std"]
    unit_scaling = pred['noise_param']['__kwargs_generative_model']['unit_scaling']
    link_temp = pred["noise_param"]["__kwargs_generative_model"]["link_temp"]
    link_shift = pred["noise_param"]["__kwargs_generative_model"]["link_shift"]

    # empirical differential expressions
    de_mat = empirical_differential_expressions(
        train_data_normalized,
        train_intv,
        ctrl=train_data_normalized[0],
        single_gene_only=False, # use all perturbations to match targets available to model
    )
    de_signs, de_pvals, _, de_targets = compute_differential_expression_sign_cached(
        train_data_normalized,
        train_data_normalized[0],
        single_gene_only=True,
        return_details=True,
        fdr_threshold=0.05,
        targets=train_intv,
    )
    assert onp.allclose(train_intv[0], 0)

    # test-set empirical differential expressions
    if test_data_normalized is not None:
        test_de_mat = empirical_differential_expressions(
            test_data_normalized,
            test_intv,
            ctrl=train_data_normalized[0],
            single_gene_only=False,  # use all perturbations to match targets available to model
        )
        test_de_signs, test_de_pvals, _, test_de_targets = compute_differential_expression_sign_cached(
            test_data_normalized,
            test_data_normalized[0],
            single_gene_only=True,
            return_details=True,
            fdr_threshold=0.05,
            targets=test_intv,
        )
    else:
        test_de_mat = None
        test_de_signs = None
        test_de_pvals = None
        test_de_targets = None

    """
    Compute causal matrices
    """
    causal_mats_kwargs = dict(
        shift=shift,
        scale=scale,
        causal_mat_kwargs={"tikhonov": tikhonov},
    )
    pred_causal_state, pred_causal_tfm = causal_mats(
        model,
        train_intv,
        param,
        **causal_mats_kwargs,
    )
    pred_causal = pred_causal_tfm

    saved = dict(
        pred_causal=pred_causal,
        var_names=var_names,
        gene_to_ensembl=gene_to_ensembl,
        de_mat=de_mat,
        de_signs_single_gene=de_signs,
        de_pvals_single_gene=de_pvals,
        de_targets_single_gene=de_targets,
        test_de_mat=test_de_mat,
        test_de_signs_single_gene=test_de_signs,
        test_de_pvals_single_gene=test_de_pvals,
        test_de_targets_single_gene=test_de_targets,
        transform_shift=shift,
        transform_scale=scale,
        rate_unit_scaling=unit_scaling,
        rate_link_temp=link_temp,
        rate_link_shift=link_shift,
    )
    saved_graph = dict(
        causal_graph=onp.array(pred_causal[list(pred_causal.keys())[-1]]["a_causal"]),
        var_names=var_names,
        gene_to_ensembl=gene_to_ensembl,
    )

    """
    Save causal matrices
    """
    save_path_str = f"{base_save_path}__tikh={tikhonov}"
    save_path = Path(save_path_str.replace(".", "_"))
    save_path.mkdir(parents=True, exist_ok=True)
    dump(saved, save_path / "causal_analysis.joblib")

    # save causal graph and annotations only
    onp.savez_compressed(save_path / f"causal_graph_{save_path.stem}.npz", **saved_graph)

    return save_path



def main():

    """
    Compute CLIPR matrices of system
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--prediction", type=str, required=True)
    parser.add_argument("--tikhonov", type=float, default=1e-2)
    kwargs = parser.parse_args()

    data_path = PROJECT_DIR / SUBDIR_RESULTS / kwargs.experiment / kwargs.data
    prediction_path = PROJECT_DIR / SUBDIR_RESULTS / kwargs.experiment / kwargs.prediction
    clipr_dump_base = PROJECT_DIR / SUBDIR_PLOTS / SUBDIR_CLIPR_DUMP

    # load data
    train_datasets, test_datasets, _, data_config, _ = load_data_cached(Path(data_path))
    train_data_normalized = train_datasets.data_normalized
    print("Loaded data from", data_path)

    # load model parameters
    pred = load_json_zip(prediction_path)
    print("Loaded model from", prediction_path)

    params = pred["theta"]
    train_intv = pred["intv_train"]
    d = pred["intv_train"].shape[-1]
    assert onp.allclose(train_datasets.intv, train_intv)

    if test_datasets is not None:
        test_data_normalized = test_datasets.data_normalized
        test_intv = pred["intv_test"]
        assert onp.allclose(test_datasets.intv, test_intv)
    else:
        test_data_normalized = None
        test_intv = None

    config = argparse.Namespace(**pred["config"])
    config.get = lambda *args: getattr(config, *args)

    # initialize model and drift function
    model = Model(
        config,
        n_envs=d, # each gene defines a perturbation (when that gene is perturbed)
        d=d,
    )

    # analyze diffusion
    fold = prediction_path.stem.rsplit("_", 1)[1]
    clipr_dump_path = clipr_dump_base / f"{prediction_path.parents[1].name}__{fold}"

    save_path_clipr = compute_and_save_clipr(
        pred,
        clipr_dump_path,
        model,
        params,
        train_data_normalized,
        train_intv,
        test_data_normalized=test_data_normalized,
        test_intv=test_intv,
        tikhonov=kwargs.tikhonov,
    )
    print("Dumped CLIPR results to:", save_path_clipr)


if __name__ == "__main__":
    main()

