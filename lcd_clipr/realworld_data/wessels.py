import warnings
import numpy as onp
import scanpy as sc
import pandas as pd
import anndata

from lcd_clipr.definitions import PATH_WESSELS
from lcd_clipr.realworld_data.helpers import \
    select_perturbations_and_genes, \
    adata_to_datasets
from lcd_clipr.realworld_data.helpers_perturb import discard_outside_perturbs


from lcd_clipr.utils.cache import init_cache
memory = init_cache()


def load_adata_raw():
    # assemble anndata object by merging 4 lanes
    adatas = []
    meta_data = pd.read_csv(PATH_WESSELS / "GSE213957_THP1-CaRPool-seq.metadata.tsv", sep='\t')

    for lane in [1, 2, 3, 4]:
        adata = sc.read_10x_mtx(PATH_WESSELS, prefix=f"THP1-CaRPool-seq_and_HEK293FTstabRNA.GEXGDO{lane}.")

        # match meta_data to cell barcodes in adata
        adata.obs["lane_barcode"] = adata.obs.index.map(lambda x: f"L{lane}_{x}")
        adata.obs = adata.obs.reset_index().set_index("lane_barcode")
        adata.obs = adata.obs.rename(columns={"index": "barcode"})
        adata.obs = meta_data.reindex(adata.obs.index)

        # discard any samples without a barcode in meta_data
        adata = adata[~adata.obs.isna().all(1)]
        adatas.append(adata)

    adata = anndata.concat(adatas, join="inner")

    # standardize gene names
    adata.var_names_make_unique("___")
    adata.var["gene_symbols"] = adata.var.index

    # remember the original gene ensembl ids
    all_ensembl_ids = [a.var.gene_ids.tolist() for a in adatas]
    assert all([all_ensembl_ids[0] == x for x in all_ensembl_ids]), "Gene IDs are not consistent across lanes"
    adata.var["gene_ids"] = adatas[0].var["gene_ids"]

    return adata


def load_adata(seed, adata_raw=None, verbose=False, **config):

    # avoid reloading the data if we already loaded it
    if adata_raw is None:
        adata = load_adata_raw()
    else:
        adata = adata_raw.copy()

    # drop variables that are not genes
    # check here ensures backward legacy compatibility with old data configs where we didn't do this
    legacy_compatibility_case = \
        type(seed) == dict \
        and "var_names" in seed \
        and any([not_gene in seed["var_names"] for not_gene in [
            "eGFP",
            "Blast",
            "Cas9",
            "Puro",
            "Cas13d",
            "AsCas12a",
            "MeCP2",
            "KRAB",
        ]])

    if not legacy_compatibility_case:
        gene_mask = adata.var["gene_ids"].map(lambda s: "ENSG" in s)
        adata._inplace_subset_var(gene_mask)

    # remember mapping from unique gene name to Ensembl IDs for later analysis
    gene_to_ensembl = dict(adata.var.gene_ids.items())

    # standardize target information
    def standardize_guide_ids(guide_id):
        gs = ["ctrl" if g == "NT" else g for g in guide_id.split("_")]
        if len(gs) == 1:
            return gs
        elif gs[0] == "ctrl":
            return gs[1]
        elif gs[1] == "ctrl":
            return gs[0]
        elif gs[0] == gs[1]:
            return gs[0]
        else:
            return ",".join(sorted((gs[0], gs[1])))

    adata.obs["guide_ids"] = pd.Categorical(adata.obs["GenePair"].map(standardize_guide_ids))

    """
    Cell quality control
    """
    # this is slow if adata.X is not a csr sparse matrix
    sc.pp.filter_cells(adata, min_genes=config["min_genes_per_cell"])
    sc.pp.filter_genes(adata, min_cells=config["min_cells_per_gene"])

    # UMI count and mitochondrial RNA quality control
    # annotate the protein-coding mitochondrial genes as `mito`
    adata.var['mito'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mito'], percent_top=None, log1p=False, inplace=True)

    # sc.pl.violin(adata, ['nCount_RNA', 'pct_counts_mito'], jitter=0.4, multi_panel=True)
    # sc.pl.scatter(adata, x='nCount_RNA', y='pct_counts_mito')

    # discard cells with high percentage of mitochondrial RNA counts (which can indicate poor cell quality/high stress)
    adata = adata[adata.obs["pct_counts_mito"] < 20]

    # other quality control based on plots
    adata = adata[adata.obs["total_counts"] > 3000, :]
    adata = adata[adata.obs["total_counts"] < 40000, :]
    adata = adata[adata.obs["n_genes_by_counts"] > 1500, :]
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
