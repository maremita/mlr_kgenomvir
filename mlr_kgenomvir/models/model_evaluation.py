import os.path
from pprint import pprint
from collections import defaultdict
import time

import numpy as np
import scipy.sparse as sp

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
        save_files=True,
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

    if save_files and os.path.isfile(clf_file) and\
            os.path.isfile(measures_file):

        if verbose:
            print("\nLoading classifier and measures from {} files".format(
                prefix+clf_name), flush=True)

        if return_coefs:
            with open(clf_file, 'rb') as fh:
                classifier = clf_load(fh)
                coeffs = classifier.coef_
                intercepts = classifier.intercept_

        with np.load(measures_file, allow_pickle=True) as f:
            measures = f['measures'].tolist()

            if verbose == 3:
                pprint(measures)

        if return_coefs:
            return coeffs, intercepts, measures
        else:
            return measures

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

    if save_files:
        with open(clf_file, 'wb') as fh:
            clf_save(classifier, fh)
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
    
    score_labels = ["X_train_sparsity", "X_test_sparsity", "fit_time",
            "n_iter", "coef_sparsity", "score_time", "train_scores", 
            "test_scores"]

    # Fetch score data to rearrange
    scores_tmp = defaultdict(list)

    for score_dict in scores_vc:
        for label in score_labels:
            if label not in ["train_scores", "test_scores"]:
                # score_dict[label] is a float or int value
                scores_tmp[label].append(score_dict[label])

            else:
                # score_dict[label] is a dict of several averages
                trorts = label.split("_")[0]
                for metriq in score_dict[label][avrg_metriq]:
                    new_label = "{}_{}_{}".format(trorts, avrg_metriq,
                            metriq)
                    scores_tmp[new_label].append(
                            score_dict[label][avrg_metriq][metriq])

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
        save_files=False,
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
            classifier.C = 1.0
        elif _lambda == 0:
            classifier.C = np.inf
        else:
            classifier.C = 1./(_lambda * n_train_samples)

    # pytorch mlr
    elif hasattr(classifier, 'alpha'):
        if penalty == "none": 
            classifier.alpha = 0.0
        else:
            classifier.alpha = _lambda * n_train_samples
    else:
        raise ValueError("Classifier does not have C or alpha attributes") 

    classifier.penalty = penalty
    if penalty == "elasticnet": classifier.l1_ratio = 0.5

    if penalty != "none":
        str_lambda = format(_lambda, '.0e') if _lambda not in list(
                range(0, 10)) else str(_lambda)
        clf_name += "_A"+str_lambda

    if learning_rate and hasattr(classifier, 'learning_rate'):
       classifier.learning_rate = learning_rate 

    ## CV iterations
    parallel = Parallel(n_jobs=n_jobs, prefer="processes", verbose=verbose)

    # cv_scores is a list of n fold dicts
    # each dict contains the results of the iteration
    cv_scores = parallel(delayed(compute_clf_coef_measures)(
        clone(classifier), clf_name+"_fold{}".format(fold),
        X_train[train_ind], X_test[test_ind], y_train[train_ind],
        y_test[test_ind], prefix, return_coefs=False,
        save_files=save_files, verbose=verbose, random_state=random_state) 
        for fold, (train_ind, test_ind) in enumerate(cv_indices))

    #print(cv_scores)
    avrg_scores = average_scores(cv_scores, average_metric)

    return avrg_scores

