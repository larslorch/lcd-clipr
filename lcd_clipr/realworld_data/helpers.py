import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from collections import defaultdict
import numpy as onp
import scanpy as sc

from lcd_clipr.core import Dataset, normalizer
from lcd_clipr.utils.gi import adata_categorize_interactions
from lcd_clipr.realworld_data.helpers_perturb import covered_genes, filter_singles, filter_doubles, select_perturbations, \
    print_perturbation_split_statistics, discard_outside_perturbs


sc.set_figure_params(dpi=300, frameon=False)


def filter_cells_genes(adata, min_cells=None, min_genes=None):
    """
    More efficient replacement for dense numpy arrays of
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    """
    if not type(adata.X) == onp.ndarray:
        warnings.warn(f"Should not call this for sparse adata.X. See docstring.")
    if min_cells is None and min_genes is None:
        return adata
    assert adata.X.ndim == 2
    msk = adata.X > 0
    if min_genes is not None:
        msk_rows = msk.sum(1) >= min_genes
        if not onp.all(msk_rows):
            # adata.X = adata.X[msk_rows]
            adata._inplace_subset_obs(msk_rows)
    if min_cells is not  None:
        msk_cols = msk.sum(0) >= min_cells
        if not onp.all(msk_cols):
            # adata.X = adata.X[:, msk_cols]
            adata._inplace_subset_var(msk_cols)
    return adata


def select_highly_variable_genes(adata, config, train_envs, test_envs, *,
                                 prioritize_perturbed,
                                 prioritize_marker):


    hvg_metrics = sc.pp.highly_variable_genes(adata, layer="log1p", flavor="seurat", inplace=False)

    # for seurat_v3: should sort "highly_variable_rank" and "acending = True"
    # however, seurat_v3 expects counts, and after library size normalization, we don't have counts anymore
    # hvg_metrics_v3 = sc.pp.highly_variable_genes(adata, n_top_genes=adata.shape[-1], flavor="seurat_v3", inplace=False)

    ordering = []
    ascending = []
    hvg_metrics["gene_name"] = adata.var_names

    # prioritize perturbed genes
    if prioritize_perturbed:
        # compute highly-variable genes *only based on the training perturbation data*
        # prioritize selection of genes perturbed in both train and test splits to be able to model
        # the genes targeted in the test perturbation split later
        perturbed_genes_train = covered_genes(train_envs, "ctrl")
        perturbed_genes_test = covered_genes(test_envs, "ctrl")

        hvg_metrics["perturbed_train"] = hvg_metrics["gene_name"].isin(perturbed_genes_train)
        hvg_metrics["perturbed_test"] = hvg_metrics["gene_name"].isin(perturbed_genes_test)

        ordering += ['perturbed_test', 'perturbed_train']
        ascending += [False, False]

    # only keep genes with minimum mean UMI per gene
    if config.get("min_mean_umi_per_gene") is not None:
        abundant = adata.X.mean(0).A.squeeze() > config["min_mean_umi_per_gene"]
        hvg_metrics["abundant"] = abundant
        ordering += ['abundant']
        ascending += [False]

    # break ties by how often the gene is selected as a marker gene
    if prioritize_marker and config["differential_expression_genes"] is not None:
        # compute differentially expressed genes (aka marker genes) for downstream analysis
        adata_marker = sc.tl.rank_genes_groups(adata, "guide_ids",
                                               n_genes=config["differential_expression_genes"],
                                               layer="log1p", reference="ctrl",
                                               rankby_abs=True,
                                               method="wilcoxon", copy=True)

        # count how often a gene is selected as a marker gene
        rank_genes_groups = adata_marker.uns["rank_genes_groups"]["names"]
        marker_gene_counter = defaultdict(int)
        for guide in rank_genes_groups.dtype.names:
            for gene in rank_genes_groups[guide].tolist(): # or :config["differential_expression_genes"] if not trimmed above
                marker_gene_counter[gene] += 1

        hvg_metrics["marker"] = onp.array([marker_gene_counter[gene] for gene in hvg_metrics["gene_name"]])

        ordering += ['marker']
        ascending += [False]

    # break ties by highly-variable gene selection score
    ordering += ['dispersions_norm']
    ascending += [False]

    # select first `n_genes` genes based on the ordering
    prio = hvg_metrics[ordering].sort_values(ordering, ascending=ascending, na_position='last').index
    hvg_metrics['selected'] = False
    hvg_metrics.loc[prio[:int(config["n_genes"])], 'selected'] = True

    # store highly-variable gene ranking for later analysis
    rank = hvg_metrics[['dispersions_norm']].sort_values('dispersions_norm', ascending=False, na_position='last').index
    hvg_metrics['hvg_rank'] = pd.Series(onp.arange(len(rank)), index=rank)
    return hvg_metrics


def select_genes(adata, config, train_envs, test_envs, guide_key, ctrl_key):

    if "n_genes" in config and config["n_genes"] is None:
        return adata.var_names.to_list(), None

    if config["gene_selection"] == "hvg_train":
        adata_train = adata[adata.obs[guide_key].isin(train_envs)]
        hvg_metrics = select_highly_variable_genes(
            adata_train, config, train_envs, test_envs,
            prioritize_perturbed=config["gene_selection_prio_perturbed"],
            prioritize_marker=config["gene_selection_prio_marker"],
        )

    elif config["gene_selection"] == "hvg_train_singles":
        adata_train_singles = adata[adata.obs[guide_key].isin([ctrl_key] + filter_singles(train_envs, ctrl_key))]
        hvg_metrics = select_highly_variable_genes(
            adata_train_singles, config, train_envs, test_envs,
            prioritize_perturbed=config["gene_selection_prio_perturbed"],
            prioritize_marker=config["gene_selection_prio_marker"],
        )

    elif config["gene_selection"] == "hvg_ctrl":
        adata_ctrl = adata[adata.obs[guide_key] == ctrl_key]
        hvg_metrics = select_highly_variable_genes(
            adata_ctrl, config, train_envs, test_envs,
            prioritize_perturbed=config["gene_selection_prio_perturbed"],
            prioritize_marker=config["gene_selection_prio_marker"],
        )

    elif isinstance(config["gene_selection"], list):
        return config["gene_selection"], None

    else:
        raise NotImplementedError(f"gene selection method {config['gene_selection']} not implemented")

    gene_sel = hvg_metrics["selected"]
    var_names = adata.var_names[gene_sel].to_list()
    return var_names, hvg_metrics


def relabeled_hvg_ranking(hvg_metrics, var_names, rank_key="hvg_rank"):
    """Relabel labeling to be dense and only for the selected genes"""
    if hvg_metrics is None:
        return {}
    selected = hvg_metrics.loc[hvg_metrics["gene_name"].isin(var_names)]
    relabeled_rank = selected[rank_key].rank(method='first').astype(int) - 1
    return dict(zip(selected['gene_name'], relabeled_rank))


def select_perturbations_and_genes(adata, config, seed, guide_key, ctrl_key, classify_gi_norman=False, verbose=False):

    """
    Perturbation selection and train-test splits
    """
    if type(seed) == dict:
        train_envs = seed["train_envs"]
        test_envs = seed["test_envs"]
    else:
        train_envs, test_envs = select_perturbations(seed, adata, config, guide_key, ctrl_key)

    assert not any([test_env in train_envs for test_env in test_envs]), "test perturbations found in train split"

    """
    Gene selection based on train split perturbations
    """
    if type(seed) == dict:
        var_names = seed["var_names"]

        # backwards compatibility
        if "hv_genes" in seed:
            hv_genes = seed["hv_genes"]
        else:
            _var_names, hvg_metrics = select_genes(adata, config, train_envs, test_envs, guide_key, ctrl_key)
            hv_genes = relabeled_hvg_ranking(hvg_metrics, var_names)
            assert set(var_names) == set(_var_names), \
                "var_names in seed and var_names computed in `select_genes` do not match"

    else:
        var_names, hvg_metrics = select_genes(adata, config, train_envs, test_envs, guide_key, ctrl_key)
        hv_genes = relabeled_hvg_ranking(hvg_metrics, var_names)

    # subset dataframe and keep consistent ordering
    gene_sel = onp.array([gene in var_names for gene in adata.var_names])
    adata._inplace_subset_var(gene_sel)
    var_names = adata.var_names.to_list()

    # filter empty cells and discard outside perturbations
    sc.pp.filter_cells(adata, min_genes=config["min_genes_per_cell"])
    adata, (train_envs, test_envs) = discard_outside_perturbs(adata, guide_key, ctrl_key, train_envs, test_envs)

    print(f"after gene selection:  {adata.shape}")
    print_perturbation_split_statistics(train=train_envs, test=test_envs, ctrl=ctrl_key)

    assert adata.shape[0] > 0 and adata.shape[1] > 0
    assert len(train_envs) > 1
    assert len(test_envs) or not config["require_test_envs"]

    """
    Remember any remaining singles not in train or test splits for later analysis and GI metrics
    """
    if type(seed) == dict:
        remaining_envs = seed["remaining_envs"]
    else:
        perturbs = adata.obs[guide_key].unique().to_list()
        perturbs.remove(ctrl_key)

        remaining_envs = [p for p in perturbs if p not in train_envs and p not in test_envs]
        remaining_envs = filter_singles(remaining_envs, ctrl_key)

    """
    Compute differentially expressed genes/marker genes for downstream analysis
    """
    if type(seed) == dict:
        marker_genes = seed["marker_genes"]
    else:
        if config["differential_expression_genes"] is None:
            marker_genes = None
        else:
            sc.tl.rank_genes_groups(adata, guide_key, layer="log1p", reference=ctrl_key, method="wilcoxon",
                                    rankby_abs=True, n_genes=config["differential_expression_genes"])
            rank_genes_groups = adata.uns["rank_genes_groups"]["names"]
            marker_genes = dict()
            for guide in rank_genes_groups.dtype.names:
                marker_genes[guide] = sorted(rank_genes_groups[guide].tolist())

            marker_genes = dict(sorted(marker_genes.items()))

    """
    Compute GI gene selection for downstream analysis
    """

    if type(seed) == dict:
        gi_genes = sorted(seed["gi_genes"])
        gi_thresholds = seed["gi_thresholds"]
        gi_genes_mean_umi = seed["gi_genes_mean_umi"]
    else:
        gi_genes_mean_umi = config["gi_genes_mean_umi"]
        gene_selection, gi_thresholds = adata_categorize_interactions(
            adata,
            save_path=config.get("save_path"),
            ctrl_key=ctrl_key,
            guide_key=guide_key,
            min_umi_count=gi_genes_mean_umi,
            classify_gi_norman=classify_gi_norman,
        )
        gi_genes = sorted([g for j, g in enumerate(var_names) if gene_selection[j]])

    # print summary statistics of dataset
    print("perturbations:")
    print(adata.obs[guide_key].value_counts())
    print(f"median cells per perturbation: {int(adata.obs[guide_key].value_counts().median())}")
    print("final dataset size:", adata.shape)

    sort_key = lambda x: (len(covered_genes(x, ctrl_key)), x)
    seeding = dict(
        train_envs=sorted(train_envs, key=sort_key),
        test_envs=sorted(test_envs, key=sort_key),
        remaining_envs=sorted(remaining_envs, key=sort_key),
        var_names=var_names,
        marker_genes=marker_genes,
        hv_genes=hv_genes,
        gi_genes=gi_genes,
        gi_thresholds=gi_thresholds,
        gi_genes_mean_umi=gi_genes_mean_umi,
    )

    return adata, seeding


def make_differential_expression_mask(marker_genes, envs, var_names):
    if marker_genes is None:
        return None
    mask = onp.zeros((len(envs), len(var_names)))
    for i, env in enumerate(envs):
        if env == "ctrl":
            continue
        if marker_genes is not None:
            try:
                for marker_gene in marker_genes[env]:
                    for j, gene in enumerate(var_names):
                        if marker_gene == gene:
                            mask[i][j] = 1
            except KeyError as e:
                print()
                warnings.warn(
                    "Perhaps `test_all_train_singles_combinations` is set to true in the config file? "
                    "In that case, `differential_expression_genes` has to be `null`, because marker genes "
                    "can only be computed for perturbations we observed in the data, so we cannot initialize "
                    "a differential expression mask here.\n"
                    "This fails with an Exception to not lead to silent bugs."
                )
                raise e

    return mask


def make_dataset_fields(adata, envs, seeding):
    if not envs:
        return None

    var_names = seeding["var_names"]
    marker_genes = seeding["marker_genes"]

    fields = defaultdict(list)
    for env in envs:
        idx = (adata.obs["guide_ids"] == env).values
        x = adata.X[idx]
        try:
            x = x.todense()
        except AttributeError:
            pass
        if x.size:
            fields['data_noisy'].append(onp.array(x))
            fields['data_normalized'].append(normalizer(onp.array(x)))
        else:
            fields['data_noisy'].append(None)
            fields['data_normalized'].append(None)
        if env == "ctrl":
            fields['intv'].append(onp.zeros(len(var_names)))
        else:
            indices = onp.array([var_names.index(g) for g in env.split(",")])
            fields['intv'].append(onp.eye(len(var_names))[indices].sum(0))

    fields = dict(**fields)
    fields['intv'] = onp.array(fields['intv'])
    fields['differential_expression_mask'] = make_differential_expression_mask(marker_genes, envs, var_names)

    return fields


def adata_to_datasets(adata, seeding):

    train_envs = seeding["train_envs"]
    test_envs = seeding["test_envs"]
    remaining_envs = seeding["remaining_envs"]

    # create data fields
    dataset_train = make_dataset_fields(adata, train_envs, seeding)
    dataset_test = make_dataset_fields(adata, test_envs, seeding)
    dataset_remaining = make_dataset_fields(adata, remaining_envs, seeding)

    return Dataset(**dataset_train), \
           Dataset(**dataset_test) if test_envs else None, \
           Dataset(**dataset_remaining) if remaining_envs else None, \
           seeding

