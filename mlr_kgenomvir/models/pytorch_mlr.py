import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.base import BaseEstimator, ClassifierMixin
#from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

import time


__author__ = "amine"


class linear_layer(nn.Module):

    def __init__(self, n_classes, n_features, bias):
        super(linear_layer, self).__init__()

        # to check dtype dependently to input data
        self.linear = nn.Linear(n_features, n_classes, bias=bias).float()

        with torch.no_grad():
            self.linear.weight.zero_()
            if bias:
                self.linear.bias.zero_()

    def forward(self, x):
        return self.linear(x)


class DataSampler():
    def __init__(self, X, Y, random_state=42):
        self.X = X
        self.Y = Y
        self.random_state=random_state
        self.size = X.shape[0]
        #
        np.random.seed(self.random_state)
        self.indices = [i for i in range(self.size)]
        np.random.shuffle(self.indices)
        self.current = 0

    def random_sample(self):
        ind = self.indices[self.current]
        self.current += 1

        if self.current == self.size:
            self.current = 0
            np.random.shuffle(self.indices)

        return self.X[ind:ind+1], self.Y[ind:ind+1]


class MLR(BaseEstimator, ClassifierMixin):

    def __init__(self, penalty='l2', tol=1e-4, alpha=1.0, l1_ratio=0., 
            learning_rate=0.001, fit_intercept=True, solver='sgd', 
            class_weight=None, max_iter=100, validation=False,
            n_iter_no_change=5, keep_losses=False, n_jobs=0,
            device='cpu', random_state=None, verbose=0):

        self.penalty = penalty
        self.tol = tol
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learning_rate=learning_rate
        self.fit_intercept = fit_intercept
        self.class_weight = class_weight # not tested yet
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.validation = validation
        self.n_iter_no_change = n_iter_no_change
        self.n_jobs = n_jobs  # not used yet
        self.device = device
        self.keep_losses = keep_losses
        self.verbose = verbose

    def fit(self, X, y):

        self.device_ = torch.device(self.device)
 
        if self.device_.type == "cuda":
            self.n_jobs = 0
 
        # TODO
        # Check validation value

        self.y_encoder = LabelEncoder()
        encoded_y = self.y_encoder.fit_transform(y).astype(np.long, copy=False)

        if not isinstance(X, torch.FloatTensor):
            X = torch.from_numpy(X).float()

        X = X.to(self.device_)
        encoded_y = torch.from_numpy(encoded_y).long().to(self.device_)

        # Store the classes seen during fit
        self.classes_ = np.unique(y)

        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        if n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 "
                    " classes in the data, but the data contains only "
                    " one class: {}".format(classes_[0]))

        # Check fit_intercept (True or False)
        _bias = self.fit_intercept 
 
        # TODO
        # Check self.class_weight
        # It has to be a Tensor
 
        # initialize a linear model
        self.model = linear_layer(n_classes, n_features, _bias).to(
                self.device_)

        # scale alpha by number of samples
        # self.alpha /= n_samples
        self.alpha_scaled_ = self.alpha / n_samples

        # Define loss function
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="mean",
                weight=self.class_weight)

        # Define optimizer
        if self.solver == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), 
                    lr=self.learning_rate, weight_decay=0)
        else:
            raise NotImplementedError("Only SGD solver is supported")
 
        # Initialize regularization attributes
        self.init_regularization()

        # Define regularizer
        self.regularizer = self._regularizer()

        # train the model
        self._fit(X, encoded_y)
 
        self.coef_ = self.model.linear.weight.data.cpu().numpy()
        self.intercept_ = self.model.linear.bias.data.cpu().numpy()

        # Return the classifier
        return self

    def _fit(self, X, y):
        n_iter = 0
        n_samples = X.shape[0]
        best_loss = np.inf
        no_improvement_count = 0

        #print(X.device, flush=True)

        if self.keep_losses: self.train_losses_ = []
        if self.validation:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.validation)
            train_ind, val_ind = next(sss.split(np.zeros(n_samples), y.cpu().numpy()))

            X_train = X[train_ind]
            y_train = y[train_ind] 
            X_val = X[val_ind]
            y_val = y[val_ind]
            #print(X_train.device, flush=True)
            X_y_loader = DataSampler(X_train, y_train, random_state=self.random_state)

            n_samples = X_train.shape[0]
            n_val_samples = X_val.shape[0]

            if self.keep_losses: self.val_losses_ = []
 
        else:
            X_y_loader = DataSampler(X, y, random_state=self.random_state)

        start = time.time()
        for epoch in range(self.max_iter):
            train_loss = torch.tensor(0.0).to(self.device_).detach()
            val_loss = None

            for i in range(n_samples):
                X_batch, y_batch = X_y_loader.random_sample()
                # Clear gradients before each batch
                self.model.zero_grad()

                # Run forward pass
                logits = self.model(X_batch)

                # compute the loss
                loss = self.cross_entropy_loss(logits, y_batch)

                # compute gradients
                loss.backward()
                train_loss += loss

                # update parameters
                self.optimizer.step()

                # Regularization (in gradient space)
                self.regularizer()

            # Check if the stopping criteria is reached
            with torch.no_grad():
                # This stopping creteria is adapted from SGD scikit-learn
                # implementation

                train_loss /= n_samples
                if self.keep_losses: self.train_losses_.append(train_loss)

                if self.validation:
                    val_logits = self.model(X_val)
                    val_loss = self.cross_entropy_loss(val_logits, y_val)

                    if self.tol > -np.inf and val_loss > best_loss - self.tol:
                        no_improvement_count += 1
                    else:
                        no_improvement_count = 0

                    if val_loss < best_loss:
                        best_loss = val_loss

                    if self.keep_losses: self.val_losses_.append(val_loss)
 
                else:
                    if self.tol > -np.inf and train_loss > best_loss - self.tol:
                        no_improvement_count += 1
                    else:
                        no_improvement_count = 0

                    if train_loss < best_loss:
                        best_loss = train_loss

                if no_improvement_count >= self.n_iter_no_change:

                    if self.verbose:
                        print("\nConvergence after {} epochs".format(epoch+1), flush=True)
                    break

                elif self.verbose == 2:
                    # predict training labels
                    y_pred = self.predict(X)
                    score = f1_score(self.y_encoder.inverse_transform(y), y_pred, average="weighted")

                    print("Epoch {}\ttrain_loss {}\tval_loss {}\t"\
                            "best_loss {}\tno_improve_count {}\tf1_score {}".format(
                                epoch+1, train_loss, val_loss,  best_loss, no_improvement_count, score), flush=True)

            n_iter +=1

        end = time.time()

        if self.verbose and n_iter >= self.max_iter:
            print("max_iter {} is reached".format(n_iter), flush=True)

        self.epoch_time_ = end - start
        self.train_loss_ = train_loss.item()
        self.n_iter_ = n_iter

    def init_regularization(self):
        # check penalty type 
        penalty_types = ['none', 'l1', 'l2', 'elasticnet']
 
        if self.penalty not in penalty_types:
            raise ValueError("Regularization type should be one of these "
                    "values {}; and got {}".format(
                        ", ".join(penalty_types), reg_type))

        if self.penalty == 'l1':
            self.l1_ratio = 1.

        elif self.penalty == 'l2':
            self.l1_ratio = 0.

        if self.penalty in ["l2", "elasticnet"]:
            l2_decay_scale = self.learning_rate*(1 - self.l1_ratio)*\
                    self.alpha_scaled_
            self.l2_decay_scale = torch.tensor(l2_decay_scale).to(
                    self.device_).detach()

        if self.penalty in ["l1", "elasticnet"]:
            l1_decay_scale = self.learning_rate*self.l1_ratio*\
                    self.alpha_scaled_
            self.l1_decay_scale = torch.tensor(l1_decay_scale).to(
                    self.device_).detach()

            w = self.model.linear.weight.data
            self.q = torch.zeros_like(w).to(self.device_).detach()
            self.wuq = torch.zeros_like(w).to(self.device_).detach()
            self.u =  torch.tensor(0.).to(self.device_).detach()

    def return_none(self):
        return None

    def _regularizer(self):

        if self.penalty == "none":
            return self.return_none

        elif self.penalty == "l1":
            return self._l1

        elif self.penalty == "l2":
            return self._l2

        elif self.penalty == "elasticnet":
            return self._elasticnet

    def _l1(self):
        """
        self.model.linear.weight.abs().sum()
        """
        w = self.model.linear.weight.data
        z = w.data.clone().to(self.device_).detach()
        lr = self.optimizer.param_groups[0]['lr']
        self.u += self.l1_decay_scale

        # w_i > 0
        self.wuq = w - (self.u + self.q)
        self.wuq.clamp_(0., np.inf)
        w.copy_(w.where(w.le(0.), self.wuq))

        # w_i < 0 
        self.wuq = w + (self.u - self.q)
        self.wuq.clamp_(-np.inf, 0.)
        w.copy_(w.where(w.ge(0.), self.wuq))

        # update q
        self.q += w - z

    def _l2(self):
        """
        self.model.linear.weight.pow(2).sum()
        """
        w = self.model.linear.weight.data
        w.add_(-self.l2_decay_scale, w)

    def _elasticnet(self):
        """
        self.model.linear.weight
        ((1 - self.l1_ratio) * _l2()) + (self.l1_ratio * _l1())
        """
        self._l2()
        self._l1()

    def decision_function(self, X):
        """
        Predict confidence scores for samples.
        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        array, shape=(n_samples, n_classes)
            Confidence scores per (sample, class) combination. 
        """

        # check_is_fitted(self)

        if not isinstance(X, torch.FloatTensor):
            X = torch.as_tensor(X, dtype=torch.float)

        X = X.to(self.device_)

        n_features = self.model.linear.weight.data.shape[1]

        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d" 
                    % (X.shape[1], n_features))

        with torch.no_grad():
            logits = self.model(X)

        return logits

    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        
        Returns
        -------
        C : array, shape [n_samples]
            Predicted class label per sample.
        """

        logits = self.decision_function(X)
        _, indices = torch.max(logits.data, 1)

        return self.classes_[indices.cpu().numpy()]

    def predict_proba(self, X):
        """
        Probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        The softmax function is used to find the predicted probability of
        each class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """

        logits = self.decision_function(X)

        return F.softmax(logits, dim=1).cpu().numpy()

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """

        logits = self.decision_function(X)

        return F.log_softmax(logits, dim=1).cpu().numpy()
