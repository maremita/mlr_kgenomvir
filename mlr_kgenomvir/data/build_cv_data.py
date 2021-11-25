from .seq_collections import SeqCollection
from .kmer_collections import GivenKmersCollection
from .kmer_collections import build_kmers_Xy_data, build_kmers
from ..utils import load_Xy_cv_data, save_Xy_cv_data

import os.path

import numpy as np
from scipy.stats import norm
from sklearn.model_selection import StratifiedShuffleSplit


__author__ = "amine"


def build_cv_data(
        seq_data,
        eval_type="CC",
        k=4,
        full_kmers=False,
        low_var_threshold=None,
        fragment_size=100,
        fragment_cov=2,
        fragment_count=1,
        sample_classes=False,
        sample_class_size_min=5,
        sample_class_size_max=200,
        sample_class_size_mean=50,
        sample_class_size_std=0,
        n_splits=3,
        test_size=0.3,
        prefix="",
        random_state=42,
        verbose=1):

    parents = []
    cv_indices = []
    ret_data = dict()

    # Sample original classes (before fragmentation if any)
    if sample_classes:
        class_sizes = list(seq_data.get_count_labels().values())
        #
        min_ = sample_class_size_min
        max_ = sample_class_size_max
        mean_ = sample_class_size_mean
        std_ = sample_class_size_std
        nb_ = len(class_sizes)
        #
        lim_fun = lambda e: min_ if e < min_ else\
                (max_ if (e > max_) else e)
        #
        sizes = list(map(lim_fun, norm.rvs(loc=mean_, 
            scale=std_, size=nb_,
            random_state=random_state).astype(np.int)))

        if verbose:
            print("\nSampling original dataset with class sizes:"\
                    "\n{}\n".format(sizes),
                    flush=True)
        #
        seq_data = seq_data.size_list_based_sample(sizes,
                seed=random_state)
        # to check
        #new_sizes = list(seq_data.get_count_labels().values())
        #print(new_sizes)

    stride = int(fragment_size/fragment_cov)

    ## Build data
    # Set dtype to int32 to comply with pytorch.
    # It is not compatible with uint64
    if eval_type in ["CC", "FF"]:

        if eval_type == "FF":
            seq_data = seq_data.extract_fragments(fragment_size,
                    stride=stride)

            if fragment_count > 1:
                seq_data = seq_data.stratified_sample(fragment_count,
                        seed=random_state)
 
        X_train, y_train = build_kmers_Xy_data(seq_data, k,
                full_kmers=full_kmers,
                low_var_threshold=low_var_threshold,
                dtype=np.int32)
        X_test = "X_train"  # a flag only
        y_test = "y_train"  # a flag only

    elif eval_type == "CF":
        # Build Train data from complete sequence
        X_train_kmer = build_kmers(seq_data, k, full_kmers=full_kmers,
                low_var_threshold=low_var_threshold, dtype=np.int32)

        X_train = X_train_kmer.data
        y_train = np.asarray(seq_data.labels)
        X_train_kmer_list = X_train_kmer.kmers_list

        # Build Test data from fragments
        seq_test = seq_data.extract_fragments(fragment_size,
                stride=stride)

        if fragment_count > 1:
            seq_test = seq_test.stratified_sample(fragment_count,
                    seed=random_state)

        parents = seq_test.get_parents_rank_list()

        X_test = GivenKmersCollection(seq_test, X_train_kmer_list,
                dtype=np.int32).data
        y_test = np.asarray(seq_test.labels)

    ## Get cross-validation indices
    sss = StratifiedShuffleSplit(n_splits=n_splits, 
            test_size=test_size, random_state=random_state)

    for train_ind, test_ind in sss.split(X_train, y_train):
        # case of CF
        if len(parents) != 0:
            test_ind = np.array(
                    [i for p in test_ind for i in parents[p]])

        cv_indices.append((train_ind, test_ind))

    ## Package data in a dictionary
    # data
    ret_data["data"] = dict()
    # I could use defaultdict, bu it's easier for serialization
    ret_data["data"]["X_train"] = X_train
    ret_data["data"]["y_train"] = y_train
    ret_data["data"]["X_test"] = X_test
    ret_data["data"]["y_test"] = y_test
    ret_data["data"]["parents"] = parents
    ret_data["data"]["cv_indices"] = cv_indices

    # Settings
    # (maybe another way to fetch arguments and add them to 
    # the dict directly)
    ret_data["settings"] = dict()
    ret_data["settings"]["eval_type"] = eval_type
    ret_data["settings"]["k"] = k
    ret_data["settings"]["full_kmers"] = full_kmers
    ret_data["settings"]["low_var_threshold"] = low_var_threshold
    ret_data["settings"]["fragment_size"] = fragment_size
    ret_data["settings"]["fragment_cov"] = fragment_cov
    ret_data["settings"]["fragment_count"] = fragment_count
    ret_data["settings"]["sample_classes"] = sample_classes
    ret_data["settings"]["sample_class_size_min"] = sample_class_size_min
    ret_data["settings"]["sample_class_size_max"] = sample_class_size_max
    ret_data["settings"]["sample_class_size_mean"] = sample_class_size_mean
    ret_data["settings"]["sample_class_size_std"] = sample_class_size_std
    ret_data["settings"]["n_splits"] = n_splits
    ret_data["settings"]["test_size"] = test_size
    ret_data["settings"]["prefix"] = prefix
    ret_data["settings"]["random_state"] = random_state

    return ret_data


def build_load_save_cv_data(
        seq_file,
        cls_file,
        prefix,
        eval_type="CC",
        k=4,
        full_kmers=False,
        low_var_threshold=None,
        fragment_size=1000,
        fragment_cov=2,
        fragment_count=1000,
        sample_classes=False,
        sample_class_size_min=5,
        sample_class_size_max=200,
        sample_class_size_mean=50,
        sample_class_size_std=0,
        n_splits=3,
        test_size=0.3,
        load_data=False,
        save_data=True,
        random_state=42,
        verbose=1):

    ## Generate the names of files
    ##############################

    Xy_cvFile = prefix+"Xy_data.npz"

    if os.path.isfile(Xy_cvFile) and load_data:
        if verbose:
            print("\nLoading data of {} with k {} from file".format(
                prefix, k), flush=True)

        data = load_Xy_cv_data(Xy_cvFile)

        if verbose: print("Done.\n", flush=True)

    else:
        if verbose:
            print("\nGenerating data of {} with k {}".format(
                prefix, k), flush=True)

        seq_data = SeqCollection((seq_file, cls_file))

        # work around to let the norm function in build_cv_data 
        # has a different random state from the norm function
        # in santasim/sim_labeled_dataset()
        if isinstance(random_state, (int)):
            random_state += 50

        data = build_cv_data(
                seq_data,
                eval_type=eval_type,
                k=k,
                full_kmers=full_kmers,
                low_var_threshold=low_var_threshold,
                fragment_size=fragment_size,
                fragment_cov=fragment_cov,
                fragment_count=fragment_count,
                sample_classes=sample_classes,
                sample_class_size_min=sample_class_size_min,
                sample_class_size_max=sample_class_size_max,
                sample_class_size_mean=sample_class_size_mean,
                sample_class_size_std=sample_class_size_std,
                n_splits=n_splits,
                test_size=test_size,
                prefix=prefix,
                random_state=random_state,
                verbose=verbose)

        if save_data:
            save_Xy_cv_data(data, Xy_cvFile)

        if verbose: print("Done.\n", flush=True)

    return data
