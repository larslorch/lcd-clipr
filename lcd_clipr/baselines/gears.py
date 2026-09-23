from gears import PertData, GEARS
from gears.utils import create_cell_graph_dataset_for_prediction

import anndata as ad
import numpy as onp
from scipy import sparse
import torch
from torch_geometric.data import DataLoader
import tempfile

from lcd_clipr.definitions import NORMAN_CELL_TYPE, REPLOGLE_CELL_TYPE, WESSELS_CELL_TYPE, SCRATCH_STORE_DIR


def gears_model_predict(model, pert_list, n_samples):
    """
    Copy of GEARS.predict function
    See https://github.com/snap-stanford/GEARS/blob/e0c27b69f8a1a611d56c3f8c5f5a168cb2cde5f6/gears/gears.py#L299
    """

    model.ctrl_adata = model.adata[model.adata.obs['condition'] == 'ctrl']
    for pert in pert_list:
        for i in pert:
            if i not in model.pert_list:
                raise ValueError(i + " is not in the perturbation graph. "
                                     "Please select from GEARS.pert_list!")

    model.best_model = model.best_model.to(model.device)
    model.best_model.eval()
    results_pred = {}

    for pert in pert_list:
        cg = create_cell_graph_dataset_for_prediction(pert, model.ctrl_adata, model.pert_list, model.device,
                                                      num_samples=n_samples)
        loader = DataLoader(cg, n_samples, shuffle=False)
        batch = next(iter(loader))
        batch.to(model.device)

        with torch.no_grad():
            p = model.best_model(batch)

        results_pred['_'.join(pert)] = p.detach().cpu().numpy()

    return results_pred


def run_gears(seed, config, train_data, train_intv, test_intv, var_names, dataset_id, *, n):

    # construct anndata object for training in the format that gears expects
    if dataset_id == "norman":
        cell_type = NORMAN_CELL_TYPE
    elif dataset_id == "replogle":
        cell_type = REPLOGLE_CELL_TYPE
    elif dataset_id == "wessels":
        cell_type = WESSELS_CELL_TYPE
    else:
        raise ValueError(f"Unknown dataset id in GEARS {dataset_id}")

    guide_ids = []
    for data, intv in zip(train_data, train_intv):
        targets = onp.where(intv == 1)[0]
        if len(targets) == 0:
            guide_ids.append(["ctrl"] * data.shape[0])
        elif len(targets) == 1:
            guide_ids.append([f"ctrl+{var_names[targets[0]]}"] * data.shape[0])
        elif len(targets) == 2:
            guide_ids.append([f"{var_names[targets[0]]}+{var_names[targets[1]]}"] * data.shape[0])
        else:
            raise ValueError("More than 2 targets are not supported by GEARS")

    x = sparse.csr_matrix(onp.matrix(onp.concatenate(train_data, axis=0)))
    adata = ad.AnnData(x)
    adata.obs["condition"] = onp.concatenate(guide_ids)
    adata.obs["cell_type"] = cell_type
    adata.var["gene_name"] = var_names

    # init scratch directory for gears downloads
    with tempfile.TemporaryDirectory(dir=SCRATCH_STORE_DIR) as scratch_dir:

        print(f"GEARS scratch dir:\n{scratch_dir}", flush=True)

        # setup dataloader
        pert_data = PertData(scratch_dir)
        pert_data.new_data_process(dataset_name=dataset_id, adata=adata)
        pert_data.load(data_path=scratch_dir + "/" + dataset_id)
        pert_data.prepare_split(split='no_test', seed=seed)
        pert_data.get_dataloader(batch_size=config["batch_size"], test_batch_size=None)

        # init model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gears_model = GEARS(pert_data,
                            device=device,
                            weight_bias_track=False,
                            proj_name='pertnet',
                            exp_name='pertnet')
        gears_model.model_initialize(
            coexpress_threshold=config["coexpress_threshold"],
            num_similar_genes_go_graph=config["num_genes_graph"],
            num_similar_genes_co_express_graph=config["num_genes_graph"],
            num_go_gnn_layers=config["gnn_layers"],
            num_gene_gnn_layers=config["gnn_layers"],
        )

        gears_model.train(epochs=config["epochs"])

        # make list of environments to make predictions for
        train_envs = [[var_names[tar] for tar in onp.where(intv == 1)[0]] for intv in train_intv]
        test_envs = [[var_names[tar] for tar in onp.where(intv == 1)[0]] for intv in test_intv]

        # print and drop perturbed genes that GEARS cannot make predictions for
        train_genes = set([gene for genes in train_envs for gene in genes])
        test_genes = set([gene for genes in test_envs for gene in genes])

        unknown_train_genes = train_genes - set(gears_model.pert_list)
        unknown_test_genes = test_genes - set(gears_model.pert_list)

        print("GEARS perturbed genes unknown:")
        print(f"train: {len(unknown_train_genes)}/{len(train_genes)}: {list(unknown_train_genes)}", flush=True)
        print(f"test:  {len(unknown_test_genes)}/{len(test_genes)}: {list(unknown_test_genes)}", flush=True)

        test_envs = [list(filter(lambda guide: guide in gears_model.pert_list, env)) for env in test_envs]

        # make predictions
        pred_dict = gears_model_predict(gears_model, test_envs, n)
        preds = onp.stack([pred_dict["_".join(env)] for env in test_envs])

        assert preds.shape == (len(test_envs), n, len(var_names)), f"{preds.shape}"

    return preds