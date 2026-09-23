from collections import defaultdict
from functools import partial
import math

import numpy as onp


def unique_genes_up_to_index(l, index):
    return len(set([gene for genes in l[:index + 1] for gene in genes.split(",")]))


def index_where_leq_val(l, val):
    return next(i for i in range(len(l) + 1) if i == len(l) or l[i] > val)


def covered_genes(envs, ctrl_key):
    if type(envs) not in [list, set, tuple]:
        envs = [envs]
    return set([gene for genes in envs for gene in genes.split(",") if genes != ctrl_key])


def filter_singles(envs, ctrl_key):
    return [env for env in envs if len(env.split(",")) == 1 and env != ctrl_key]


def filter_doubles(envs, ctrl_key):
    return [env for env in envs if len(env.split(",")) == 2 and env != ctrl_key]


def filter_doubles_seen_k(envs, gene_set, k, ctrl_key):
    return [env for env in envs if len(env.split(",")) == 2 and env != ctrl_key
            and sum(guide in gene_set for guide in env.split(",")) == k]


def select_perturbations(seed, adata, config, guide_key, ctrl_key):
    """
    Select random perturbations for training, validation, and testing
    assuming that we require the test set to contain combinations of perturbations
    observed in the training set

    Note: we limit the overall number of perturbations if we model less genes than targeted across all perturbations
    to avoid violating the above assumption after subselecting genes and discarding perturbations outside the gene set
    """
    rng = onp.random.default_rng(seed)

    # randomize perturbation ordering based on seed to make subsequent split random
    all_perturbs = adata.obs[guide_key].unique().to_list()
    all_perturbs.remove(ctrl_key)
    all_perturbs = rng.permutation(all_perturbs).tolist()

    singles = filter_singles(all_perturbs, ctrl_key)
    doubles = filter_doubles(all_perturbs, ctrl_key)

    print("perturbations")
    print("singles:", len(singles))
    print("doubles:", len(doubles))
    print()

    # by default all singles are in the train set
    train_singles = singles[:config.get("perturbations_train_singles_max") or len(singles)]
    test_singles = []

    """
    Split doubles into validation + test sets
    """
    # subset all doubles such that the total number of genes targeted is at most `n_genes`
    # so that we do not select a perturbation we may later discard (only happens when `n_genes` is small)
    # this only affects case where `n_genes` is small in local debugging settings
    doubles_cumul_genes = list(map(partial(unique_genes_up_to_index, doubles), range(len(doubles))))
    doubles_index_num_genes_covered = index_where_leq_val(doubles_cumul_genes, config.get("n_genes") or float("inf"))
    doubles = doubles[:doubles_index_num_genes_covered]

    assert config["n_perturbations_val"] <= len(doubles)

    val_cutoff = config["n_perturbations_val"]
    val_doubles = doubles[:val_cutoff]
    test_doubles = doubles[val_cutoff:]

    print("split into val and test ")
    print("doubles val:", len(val_doubles))
    print("doubles test:", len(test_doubles))
    print()

    assert not set(val_doubles) & set(test_doubles)

    """
    Select train and test set based on fold and config
    """
    if not config["require_test_envs"]:
        train_doubles = val_doubles + test_doubles
        test_doubles = []
        assert len(train_doubles) + len(test_doubles) == len(doubles)

    elif "n_folds" in config:
        assert "fold" in config, "`fold` is specified by ExperimentManager"

        if config["fold"] == "val":
            # validation never sees the test set
            # train on half of doubles and validate on other half
            del test_doubles

            val_mid = math.floor(len(val_doubles) * 0.5)
            test_doubles = val_doubles[val_mid:]
            train_doubles = val_doubles[:val_mid]

            assert len(train_doubles) + len(test_doubles) == len(val_doubles)

        else:
            # doubles: test on one fold and train on all other folds including the validation set
            test_splits = onp.array_split(test_doubles, config["n_folds"])
            test_splits = list(map(lambda arr: arr.tolist(), test_splits))

            assert 1 <= config["fold"] <= config["n_folds"]
            fold_idx = config["fold"] - 1

            test_doubles = test_splits[fold_idx]
            train_doubles = [
                double
                for doubles in [
                    val_doubles,
                    *test_splits[:fold_idx],
                    *test_splits[fold_idx + 1:]
                ]
                for double in doubles
            ]

            assert len(train_doubles) + len(test_doubles) == len(doubles)

            # singles: only test on a singles split if test_singles explicitly provided in config
            if "perturbations_test_singles" in config:
                # randomize and remove provided test singles from train singles
                test_singles = config["perturbations_test_singles"]
                test_singles = rng.permutation(test_singles).tolist()
                for single in test_singles:
                    assert single in train_singles, f"provided test single {single} not in train_singles"
                train_singles = [single for single in train_singles if single not in test_singles]

                # split provided test singles into folds
                test_splits_singles = onp.array_split(test_singles, config["n_folds"])
                test_splits_singles = list(map(lambda arr: arr.tolist(), test_splits_singles))

                test_singles = test_splits_singles[fold_idx]
                train_singles = [
                    single
                    for singles in [
                        train_singles,
                        *test_splits_singles[:fold_idx],
                        *test_splits_singles[fold_idx + 1:]
                    ]
                    for single in singles
                ]

                assert len(train_singles) + len(test_singles) == len(singles)

    else:
        assert "fold" not in config
        train_doubles = test_doubles
        test_doubles = val_doubles

        assert len(train_doubles) + len(test_doubles) == len(doubles)

    del val_doubles
    assert not set(train_doubles) & set(test_doubles)

    # make sure max number of doubles is satisfied
    train_doubles = train_doubles[:config.get("perturbations_train_doubles_max") or len(train_doubles)]

    print("train+test split")
    print("preliminary")
    print("singles train:", len(train_singles))
    print("singles test:", len(test_singles))
    print("doubles train:", len(train_doubles))
    print("doubles test:", len(test_doubles))
    print()

    """
    Discard test set doubles that are not fully covered in the training set
    """
    train_covered = covered_genes(train_singles + train_doubles, ctrl_key)
    test_doubles = [
        env for env in test_doubles
        if all(gene in train_covered for gene in covered_genes(env, ctrl_key))
    ]

    print("train+test split")
    print("after discarding test doubles not fully covered in train")
    print("singles train:", len(train_singles))
    print("singles test:", len(test_singles))
    print("doubles train:", len(train_doubles))
    print("doubles test:", len(test_doubles))
    print()

    assert len(test_singles) or len(test_doubles) or not config["require_test_envs"], \
        "No test singles or doubles left after filtering." \
        "Try decreasing the test set size or increasing the max number of doubles in the train set."

    train_envs = [ctrl_key] + train_singles + train_doubles
    test_envs = test_singles + test_doubles

    assert test_envs or not config["require_test_envs"]
    assert not set(train_envs) & set(test_envs)

    print("perturbations after train-test split")
    print_perturbation_split_statistics(train=train_envs, test=test_envs, ctrl=ctrl_key)
    print()
    
    """
    Add all doubles to the test set for which we have both genes in the train set
    """
    if config.get("test_all_train_singles_combinations", False):
        train_genes = covered_genes(train_envs, ctrl_key)
        for a in train_genes:
            for b in train_genes:
                if a < b:
                    double = f"{a},{b}"
                    if double not in train_envs and double not in test_envs:
                        test_envs.append(double)

        print("perturbations after adding all possible doubles to test set")
        print_perturbation_split_statistics(train=train_envs, test=test_envs, ctrl=ctrl_key)
        print()

        assert not set(train_envs) & set(test_envs)

    return train_envs, test_envs


def print_perturbation_statistics(name, envs, seen_genes, ctrl_key):
    print(f"{name}  "
          f"singles: {len(filter_singles(envs, ctrl_key)):3d}  "
          f"doubles: {len(filter_doubles(envs, ctrl_key)):3d}  "
          f"(seen none: {len(filter_doubles_seen_k(envs, seen_genes, 0, ctrl_key))}, "
          f"seen one: {len(filter_doubles_seen_k(envs, seen_genes, 1, ctrl_key))}, "
          f"seen two: {len(filter_doubles_seen_k(envs, seen_genes, 2, ctrl_key))})  "
          f"genes targeted: {len(covered_genes(envs, ctrl_key)):3d}",
          flush=True)
    return


def print_perturbation_split_statistics(*, train, ctrl, val=None, test=None):

    train_genes = covered_genes(train, ctrl)

    print_perturbation_statistics("train ", train, train_genes, ctrl)

    if val is not None:
        print_perturbation_statistics("val   ", val, train_genes, ctrl)

    if test is not None:
        print_perturbation_statistics("test  ", test, train_genes, ctrl)

    if val is not None and test is not None:
        unknown_val_genes = covered_genes(val, ctrl) - train_genes
        unknown_test_genes = covered_genes(test, ctrl) - train_genes
        unknown_overlap = unknown_val_genes & unknown_test_genes
        prop_unknown_test = len(unknown_test_genes - unknown_overlap) / len(unknown_test_genes) if unknown_test_genes else 0

        print(f"val-test unk genes overlap: {len(unknown_overlap)} "
              f"(val unk: {len(unknown_val_genes)}, test unk: {len(unknown_test_genes)}, "
              f"test unk after train+val: {prop_unknown_test:.3f}) ", flush=True)

    all_envs_genes = covered_genes(train + (val or []) + (test or []), ctrl)
    print(f"genes targeted train:      {len(train_genes):3d}", flush=True)
    if val is None and test is not None:
        print(f"genes targeted train+test: {len(all_envs_genes):3d}", flush=True)
    if val is not None and test is not None:
        print(f"genes targeted train+val+test: {len(all_envs_genes):3d}", flush=True)

    return


def discard_outside_perturbs(adata, guide_key, ctrl_key, *envs):
    """
    Discard perturbations for which a targeted gene was not measured in the dataset
    """
    is_measured = lambda guide: all([g in adata.var_names for g in covered_genes(guide, ctrl_key)])

    # adata
    adata_guides_measured = list(filter(is_measured, adata.obs[guide_key].unique()))
    adata = adata[adata.obs[guide_key].isin(adata_guides_measured), :]
    assert all(map(is_measured, adata.obs[guide_key].unique()))

    if not envs:
        return adata

    # envs
    filtered_envs = []
    for env_list in envs:
        env_guides_measured = list(filter(is_measured, env_list))
        filtered_envs.append(env_guides_measured)

    return adata, filtered_envs
