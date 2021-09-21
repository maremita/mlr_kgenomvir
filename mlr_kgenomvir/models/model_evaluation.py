import os.path
from pprint import pprint
from collections import defaultdict
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.base import clone
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from joblib import Parallel, delayed

import joblib
import torch

__author__ = "amine"


def compute_clf_coef_measures(
        classifier,
        clf_name,
        X_train,
        X_test,
        y_train,
        y_test,
        prefix,
        return_coefs=True,
        save_model=False,
        save_result=False,
        verbose=0,
        random_state=42):

    measures = dict()

    if "torch" in classifier.__module__:
        clf_ext = ".pt"
        clf_load = torch.load
        clf_save = torch.save

    elif "sklearn" in classifier.__module__:
        clf_ext = ".jb"
        clf_load = joblib.load
        clf_save = joblib.dump

    else:
        raise ValueError("Classifier type is not known: {}".format(
            type(classifier))) 

    clf_file = prefix+clf_name+clf_ext
    measures_file = prefix+clf_name+".npz"

    ## Try to load the model and results from files
    ## ############################################
    # Load model
    load_model = False
    if os.path.isfile(clf_file) and save_model and return_coefs:

        if verbose:
            print("\nLoading classifier from {} file".format(
                clf_file), flush=True)

        with open(clf_file, 'rb') as fh:
            classifier = clf_load(fh)
            coeffs = classifier.coef_
            intercepts = classifier.intercept_
            load_model = True

    # Load results
    if os.path.isfile(measures_file) and save_result:

        if verbose:
            print("\nLoading measures from {} file".format(
                measures_file), flush=True)

        with np.load(measures_file, allow_pickle=True) as f:
            measures = f['measures'].tolist()

            if verbose == 3:
                pprint(measures)

        if load_model:
            return coeffs, intercepts, measures
        else:
            return measures

    ## Compute the model and the results
    ## #################################
    classifier.model_name = clf_name
    measures['model_name'] = clf_name

    if sp.issparse(X_train):
        measures['X_train_sparsity'] = np.mean(X_train.todense().ravel() == 0)
        measures['X_test_sparsity'] = np.mean(X_test.todense().ravel() == 0)
    else:
        measures['X_train_sparsity'] = np.mean(X_train.ravel() == 0)
        measures['X_test_sparsity'] = np.mean(X_test.ravel() == 0)

    ## Compute classifier coefficients and performance
    ##################################################
    if verbose:
        print("\nTrain-test the model {}".format(clf_name), flush=True)

    # Train classifier
    start = time.time()
    classifier.fit(X_train, y_train)
    end = time.time()
    measures["fit_time"] = end - start
 
    coeffs = classifier.coef_
    intercepts = classifier.intercept_
 
    measures['coef_sparsity'] = np.mean(coeffs.ravel() == 0)

    if hasattr(classifier, 'n_iter_'):
        if isinstance(classifier.n_iter_, (int, float, np.integer)):
            n_iter = classifier.n_iter_
 
        elif isinstance(classifier.n_iter_, np.ndarray): 
            n_iter = classifier.n_iter_.tolist()[0]

        measures['n_iter'] = n_iter

    if hasattr(classifier, 'train_loss_'):
        measures['train_loss'] = classifier.train_loss_

    if hasattr(classifier, 'best_loss_'):
        measures['best_loss'] = classifier.best_loss_

    if hasattr(classifier, 'train_losses_'):
        measures['train_losses'] = classifier.train_losses_

    if hasattr(classifier, 'val_losses_'):
        measures['val_losses'] = classifier.val_losses_

    if hasattr(classifier, 'epoch_time_'):
        measures['epoch_time'] = classifier.epoch_time_

    # Prediction on train
    y_train_pred = classifier.predict(X_train)

    # Prediction on test
    start = time.time()
    y_pred = classifier.predict(X_test)
    end = time.time() 
    measures["score_time"] = end - start

    # labels
    measures["labels"] = dict()
    measures["labels"]["y_test"] = y_test.tolist()
    measures["labels"]["y_pred"] = y_pred.tolist()

    # Score metrics
    measures["train_scores"] = dict()
    measures["test_scores"] = dict()
    score_names = ["precision", "recall", "fscore", "support"]

    for average in ["micro", "macro","weighted"]:
        measures["train_scores"][average] = dict()
        measures["test_scores"][average] = dict()
 
        # scores on train (evaluate under/over-fitting)
        tr_scores = precision_recall_fscore_support(y_train, y_train_pred,
                average=average)

        # scores on test
        ts_scores = precision_recall_fscore_support(y_test, y_pred, 
                average=average)

        for name, tr, ts in zip(score_names, tr_scores, ts_scores):
            if name != "support":
                measures["train_scores"][average][name] = tr
                measures["test_scores"][average][name] = ts

    report = classification_report(y_test, y_pred, output_dict=True)
    measures["report"] = report 

    if save_model:
        with open(clf_file, 'wb') as fh:
            clf_save(classifier, fh)

    if save_result:
        np.savez(measures_file, measures = measures)

    if verbose == 3:
        pprint(measures)
 
    if return_coefs:
        return coeffs, intercepts, measures

    else:
        return measures


def average_scores(scores_vc, avrg_metriq):
    n_folds = len(scores_vc)
    moyennes = dict()
 
    # Data that we need to fetch and average
    # X_train_sparsity: float
    # X_test_sparsity: float 
    # fit_time: float 
    # n_iter: int
    # coef_sparsity : float 
    # score_time : float 
    # train_scores: dict
    # test_scores: dict
    # train_loss: float
 
    score_labels = ["X_train_sparsity", "X_test_sparsity", "fit_time",
            "n_iter", "coef_sparsity", "score_time", "train_scores", 
            "test_scores", "train_loss"]

    # Fetch score data to rearrange
    scores_tmp = defaultdict(list)

    for score_dict in scores_vc:
        for label in score_labels:
            if label in score_dict:
                if label in ["train_scores", "test_scores"]:
                    # score_dict[label] is a dict of several averages
                    trorts = label.split("_")[0]
                    for metriq in score_dict[label][avrg_metriq]:
                        new_label = "{}_{}_{}".format(trorts, avrg_metriq,
                                metriq)
                        scores_tmp[new_label].append(
                                score_dict[label][avrg_metriq][metriq])
                else: 
                    # score_dict[label] is a float or int value
                    scores_tmp[label].append(score_dict[label])

    # Compute mean and std
    mean_std = dict() 
    for label in scores_tmp:
        values = np.array(scores_tmp[label])
        mean_std[label] = [values.mean(), values.std()]

    #print(mean_std)
    return mean_std


def perform_mlr_cv(
        classifier,
        clf_name,
        penalty,
        _lambda,
        train_test_data,
        prefix,
        learning_rate=None,
        metric="fscore",
        average_metric="weighted",
        n_jobs=1,
        save_model=False,
        save_result=False,
        verbose=0,
        random_state=42):

    ## Get data
    X_train = train_test_data["X_train"]
    y_train = train_test_data["y_train"]
    X_test = X_train
    y_test = y_train

    if isinstance(train_test_data["X_test"], (np.ndarray, np.generic)):
        X_test = train_test_data["X_test"]
        y_test = train_test_data["y_test"]

    cv_indices = train_test_data["cv_indices"]

    ## Compute C of MLR
    n_train_samples = cv_indices[0][0].shape[0] 

    ## MLR hyper-parameters
    # scikit learn mlr
    if hasattr(classifier, 'C'):
        if penalty == "none":
            # To avoid sklearn warning (_logistic.py#L1504)
            classifier.C = 1.0
        elif _lambda == 0:
            classifier.C = np.inf
        else:
            classifier.C = 1./_lambda

    # pytorch mlr
    elif hasattr(classifier, 'alpha'):
        if penalty == "none": 
            classifier.alpha = 0.0
        else:
            classifier.alpha = _lambda
    else:
        raise ValueError("Classifier does not have C or alpha attributes") 

    classifier.penalty = penalty

    if penalty == "elasticnet": classifier.l1_ratio = 0.5

    if hasattr(classifier, 'learning_rate'):
        _lr = classifier.learning_rate

        if learning_rate:
            classifier.learning_rate = learning_rate
            _lr = learning_rate

        # add learning rate to clf name 
        str_lr = format(_lr, '.0e') if _lr not in list(
                range(0, 10)) else str(_lr)
        clf_name += "_LR"+str_lr

    # add lambda to clf name
    if penalty != "none":
        str_lambda = format(_lambda, '.0e') if _lambda not in list(
                range(0, 10)) else str(_lambda)
        clf_name += "_A"+str_lambda

    ## CV iterations
    parallel = Parallel(n_jobs=n_jobs, prefer="processes", verbose=verbose)

    # cv_scores is a list of n fold dicts
    # each dict contains the results of the iteration
    cv_scores = parallel(delayed(compute_clf_coef_measures)(
        clone(classifier), clf_name+"_fold{}".format(fold),
        X_train[train_ind], X_test[test_ind], y_train[train_ind],
        y_test[test_ind], prefix, return_coefs=False,
        save_model=save_model, save_result=save_result, verbose=verbose,
        random_state=random_state) 
        for fold, (train_ind, test_ind) in enumerate(cv_indices))

    #print(cv_scores)
    avrg_scores = average_scores(cv_scores, average_metric)

    return avrg_scores


def get_mlr_cv_from_files(
        clf_name,
        penalty,
        _lambda,
        prefix,
        folds,
        learning_rate=None,
        metric="fscore",
        average_metric="weighted",
        verbose=0):

    if learning_rate:
        _lr = learning_rate

        # add learning rate to clf name 
        str_lr = format(_lr, '.0e') if _lr not in list(
                range(0, 10)) else str(_lr)
        clf_name += "_LR"+str_lr

    # add lambda to clf name
    if penalty != "none":
        str_lambda = format(_lambda, '.0e') if _lambda not in list(
                range(0, 10)) else str(_lambda)
        clf_name += "_A"+str_lambda

    cv_scores = list()

    for fold in range(folds):

        measures_file = prefix+clf_name+"_fold{}.npz".format(fold)
        # Load results
        if os.path.isfile(measures_file):
            if verbose:
                print("\nLoading measures from {} file".format(
                    measures_file), flush=True)

            with np.load(measures_file, allow_pickle=True) as f:
                measures = f['measures'].tolist()
                cv_scores.append(measures)
        else:
            raise FileNotFoundError(measures_file)

    avrg_scores = average_scores(cv_scores, average_metric)

    return avrg_scores


def compile_score_names(eval_metric, avrg_metric):
    names = []
    if eval_metric == "all":
        names = ["test_{}_precision".format(avrg_metric), 
                "test_{}_recall".format(avrg_metric), 
                "test_{}_fscore".format(avrg_metric)]
    else:
        names = ["train_{}_{}".format(avrg_metric, eval_metric),
                "test_{}_{}".format(avrg_metric, eval_metric)]

    names.extend(["coef_sparsity", "convergence", "X_train_sparsity", "train_loss"])

    return names


def make_clf_score_dataframes(clf_covs, rows, nom_scores, max_iter):
    # rows are main evaluation hyperparams (coverage|lambda|k|learning_rate) 
    # columns are score_names 
    df_scores = defaultdict(dict)

    for clf_name in clf_covs:
        df_mean = pd.DataFrame(index=rows, columns=nom_scores)
        df_std = pd.DataFrame(index=rows, columns=nom_scores)

        for row in rows:
            scores = clf_covs[clf_name][row]
            for score_name in nom_scores:
                if score_name in scores:
                    df_mean.loc[row, score_name] = scores[score_name][0]
                    df_std.loc[row, score_name] = scores[score_name][1]

                elif score_name == "convergence":
                    df_mean.loc[row,score_name] = scores["n_iter"][0]/max_iter
                    df_std.loc[row, score_name] = scores["n_iter"][1]/max_iter

        df_scores[clf_name]["mean"] = df_mean.astype(np.float)
        df_scores[clf_name]["std"] = df_std.astype(np.float)

    return df_scores


def average_scores_dataframes(list_scores):
    """
    The function compute the mean values of several dataframes.
    The dataframes are distributed by classifier type and contain 
    mean and std scores.
    The function is used in simulation-based evaluation scripts
    """
    # The std of scores must be converted 
    # to variance before computing their means as in:
    # https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation

    # list_scores: [{"clf_L1": {"mean": df, "std": df}, ..}, ..]
    # Dataframes have the same rows and columns.
    # Each element in list_scores corresponds to the result of an iteration
    # of the simulation and its produced by make_clf_score_dataframes function.

    df_scores = defaultdict(dict)
    df_means = defaultdict(pd.DataFrame)
    df_vars = defaultdict(pd.DataFrame)

    clf_names = list_scores[0].keys() # get names of classifiers

    for i in range(len(list_scores)):
        for clf in clf_names:
            df_means[clf] = pd.concat([df_means[clf], list_scores[i][clf]["mean"]])
            df_vars[clf] = pd.concat([df_vars[clf], list_scores[i][clf]["std"]**2])

    for clf in clf_names:
        df_scores[clf]["mean"] = df_means[clf].groupby(level=0).mean()
        df_scores[clf]["std"] = df_vars[clf].groupby(level=0).mean().apply(np.sqrt)

    return df_scores


def plot_cv_figure(scores, score_labels, x_values, xlabel,  out_file):
    fig_format = "png"
    #fig_format = "eps"
    fig_dpi = 150

    fig_file = out_file+"."+fig_format
    fig_title = os.path.splitext((os.path.basename(fig_file)))[0]
    
    nb_clfs = len(scores)

    cmap = cm.get_cmap('tab20')
    colors = [cmap(j/20) for j in range(0,20)]

    styles = ["s-","o-","d-.","^-.","x-","h-","<-",">-","*-","p-"]
    sizefont = 12

    f, axs = plt.subplots(1, nb_clfs, figsize=(8*nb_clfs, 5))

    plt.rcParams.update({'font.size':sizefont})
    plt.subplots_adjust(wspace=0.12, hspace=0.1)

    line_scores = [l for l in score_labels if "X" not in l and l != "train_loss"]
    area_scores = [l for l in score_labels if "X" in l] 

    ind = 0
    for i_c, classifier in enumerate(scores):
        df_mean = scores[classifier]["mean"]
        df_std = scores[classifier]["std"]

        dfl_mean = df_mean[line_scores]
        dfl_std = df_std[line_scores]

        dfa_mean = df_mean[area_scores]
        dfa_std = df_std[area_scores]

        p = dfl_mean.plot(kind='line', ax=axs[ind], style=styles, 
                fontsize=sizefont, markersize=8)

        dfa_mean.plot(kind='area', ax=axs[ind], alpha=0.2, color="gray",
                fontsize=sizefont)

        # For ESP transparent rendering
        p.set_rasterization_zorder(0)

        xticks = np.array([j for j in range(len(x_values))])
 
        p.set_title(classifier)
        p.set_xticks(xticks)
        p.set_xticklabels(x_values, fontsize=sizefont)

        for x_v, x_t in zip(x_values, p.get_xticklabels()):
            if "e" in x_v or len(x_v) >= 4: 
                x_t.set_rotation(45)

        p.set_ylim([-0.05, 1.05])
        p.set_xlabel(xlabel, fontsize=sizefont+1) # 'Coverage'

        zo = -ind
        for score_name in dfl_mean:
            m = dfl_mean[score_name]
            s = dfl_std[score_name]

            p.fill_between(xticks, m-s, m+s, alpha=0.1, zorder=zo)
            zo -= 1

        p.get_legend().remove()
        p.grid()
        ind += 1
    
    # print legend for the last subplot
    p.legend(loc='upper left', fancybox=True, shadow=True, 
            bbox_to_anchor=(1.01, 1.02))

    plt.suptitle(fig_title)
    plt.savefig(fig_file, bbox_inches="tight",
            format=fig_format, dpi=fig_dpi)

