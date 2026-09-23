import numpy as onp
import scanpy as sc
from lcd_clipr.definitions import PATH_REPLOGLE
from lcd_clipr.realworld_data.helpers import \
    filter_cells_genes, \
    select_perturbations_and_genes, \
    adata_to_datasets
from lcd_clipr.realworld_data.helpers_perturb import discard_outside_perturbs


def load_adata(seed, adata_raw=None, verbose=False, **config):

    # avoid reloading the data if we already loaded it
    if adata_raw is None:
        adata = sc.read_h5ad(PATH_REPLOGLE).copy()
    else:
        adata = adata_raw.copy()

    # rename columns to avoid confusion with variable names
    adata.obs.rename(columns={"gene": "guide_ids"}, inplace=True)
    adata.obs["guide_ids"].replace("non-targeting", "ctrl", inplace=True)

    # use gene symbols as variable names (even though the current variable names are their RNA transcripts)
    adata.var["gene_name"] = adata.var["gene_name"].astype(str)
    adata.var = adata.var.reset_index().set_index("gene_name")

    # genes sometimes have more than one transcript (e.g. due to alternative splicing?)
    # hence naming each RNA by its gene symbol may not be unique in some cases
    adata.var_names_make_unique("___")

    # remember mapping from unique gene name to Ensembl IDs for later analysis
    gene_to_ensembl = dict(adata.var.gene_id.items())

    """
    Cell quality control
    This has partially already been performed by the authors, so this is a sanity check
    """
    # this is slow if adata.X is not a csr sparse matrix
    # sc.pp.filter_cells(adata, min_genes=config["min_genes_per_cell"])
    # sc.pp.filter_genes(adata, min_cells=config["min_cells_per_gene"])
    adata = filter_cells_genes(adata, min_genes=config["min_genes_per_cell"], min_cells=config["min_cells_per_gene"])

    # mitochondrial RNA quality control
    # annotate the protein-coding mitochondrial genes as `mito`
    adata.var['mito'] = adata.var_names.str.startswith('MT-')

    # compute count metrics
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mito'], percent_top=None, log1p=False, inplace=True)

    # discard cells with high percentage of mitochondrial RNA counts (which can indicate poor cell quality/high stress)
    adata = adata[adata.obs["pct_counts_mito"] < 22]

    # other quality control based on plots
    adata = adata[adata.obs["total_counts"] > 3000, :]
    adata = adata[adata.obs["n_genes_by_counts"] < 6500, :]

    # library size normalization for gene selection
    ctrl = adata[adata.obs["guide_ids"] == "ctrl"].X
    ctrl_median_libsize = onp.median(ctrl.sum(1), axis=0).item()
    adata = adata.copy()
    x_norm = sc.pp.normalize_total(adata, target_sum=ctrl_median_libsize, inplace=False)['X']
    adata.layers["log1p"] = sc.pp.log1p(x_norm, copy=True)
    assert adata.shape[0] > 0 and adata.shape[1] > 0, "No cells or genes left after filtering cells"

    """
    Perturbation quality control
    """
    # discard perturbations targeting genes that are not measured
    adata = discard_outside_perturbs(adata, "guide_ids", "ctrl")

    # only consider perturbations with a significant number of cells measured
    counts = adata.obs["guide_ids"].value_counts()
    all_envs = counts[counts >= config["min_cells_per_perturbation"]].index.to_list()
    adata = adata[adata.obs["guide_ids"].isin(all_envs)]

    assert adata.shape[0] > 0 and adata.shape[1] > 0, "No cells or genes left after filtering perturbations"


    """
    Perturbation selection and train-test splits
    Gene selection
    Differential expression analysis
    """
    adata, seeding = select_perturbations_and_genes(adata, config, seed, "guide_ids", "ctrl",
                                                    classify_gi_norman=False, verbose=verbose)

    seeding["gene_to_ensembl"] = gene_to_ensembl

    return adata, seeding


def load(seed, config, adata_raw=None, verbose=False):
    adata, seeding = load_adata(seed, adata_raw=adata_raw, verbose=verbose, **config)
    return adata_to_datasets(adata, seeding)
