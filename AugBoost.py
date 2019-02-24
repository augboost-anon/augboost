from sklearn.ensemble.gradient_boosting import BaseGradientBoosting, GradientBoostingClassifier, \
    GradientBoostingRegressor
from sklearn.preprocessing.data import QuantileTransformer
from sklearn.ensemble.gradient_boosting import VerboseReporter
from sklearn.ensemble.gradient_boosting import _random_sample_mask
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.ensemble._gradient_boosting import predict_stage

from sklearn.utils import check_random_state
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils import column_or_1d
from sklearn.utils import check_consistent_length
from sklearn.utils.multiclass import check_classification_targets

from sklearn.tree.tree import DTYPE
from sklearn.metrics import log_loss

from scipy.sparse import issparse
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix

import numpy as np
import pandas as pd
import pickle
import datetime
import matplotlib.pyplot as plt

from AugmentationUtils import get_transformed_matrix, get_transformed_params


class AugBoostBase(BaseGradientBoosting):

    def __init__(self, loss, learning_rate, n_estimators, criterion,
                 min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                 max_depth, min_impurity_decrease, min_impurity_split,
                 init, subsample, max_features,
                 random_state, n_features_per_subset, max_epochs, is_classification,
                 experiment_name='', augmentation_method='nn', trees_between_feature_update=10,
                 save_mid_experiment_accuracy_results=False, warm_start=False, presort='auto',
                 validation_fraction=0.1, n_iter_no_change=None, tol=1e-4, max_leaf_nodes=None,):

        super(AugBoostBase, self).__init__(loss, learning_rate, n_estimators, criterion,
                                      min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                                      max_depth, min_impurity_decrease, min_impurity_split,
                                      init, subsample, max_features, random_state, max_leaf_nodes,
                                           warm_start, presort, max_epochs)
        self.n_features_per_subset = n_features_per_subset
        self.is_classification = is_classification
        self.experiment_name = experiment_name
        self.augmentation_method = augmentation_method
        self.trees_between_feature_update = trees_between_feature_update
        self.save_mid_experiment_accuracy_results = save_mid_experiment_accuracy_results
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.warm_start = warm_start
        self.tol = tol
        self.alpha = 0.9
        self.max_epochs = max_epochs
        self.max_leaf_nodes = max_leaf_nodes
        self._SUPPORTED_LOSS = ('deviance', 'exponential', 'ls', 'lad', 'huber', 'quantile')

    def _predict_stages(self, X):
        estimators = self.estimators_
        scale = self.learning_rate

        n_estimators = estimators.shape[0]
        K = estimators.shape[1]

        X_original = X
        X_normed = self.normalizer.transform(X)

        output = self._init_decision_function(np.concatenate([X_original, X_original], axis=1))

        if issparse(X):
            if X.format != 'csr':
                raise ValueError("When X is a sparse matrix, a CSR format is"
                                 " expected, got {!r}".format(type(X)))
        else:
            if not isinstance(X, np.ndarray) or np.isfortran(X):
                raise ValueError("X should be C-ordered np.ndarray,"
                                 " got {}".format(type(X)))
            for i in range(len(self.augmentations_)):
                X = get_transformed_matrix(X_normed, self.augmentations_[i], augmentation_method=self.augmentation_method)
                all_features_temp = np.concatenate([X_original, X], axis=1)
                tree_preds_to_add = np.zeros(output.shape)
                for k in range(K):
                    tree = estimators[i, k].tree_
                    predictions_temp = tree.predict(all_features_temp)
                    tree_preds_to_add[:, k] = scale * predictions_temp[:, 0]
                output += tree_preds_to_add
        return output

    def _init_state(self):
        self.augmentations_ = []
        self.normalizer = QuantileTransformer(n_quantiles=1000, random_state=77)
        self.val_score_ = np.zeros((int(self.n_estimators),), dtype=np.float64)
        self.nn_histories_ = []
        super(AugBoostBase,self)._init_state()

    def _clear_state(self):
        if hasattr(self, 'val_score_'):
            del self.val_score_
        if hasattr(self, 'nn_histories_'):
            del self.nn_histories_
        if hasattr(self, '_rng'):
            del self._rng
        super(AugBoostBase,self)._clear_state()

    def _resize_state(self):
        self.val_score_.resize(self.n_estimators)
        super(AugBoostBase, self)._resize_state()

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None, monitor=None):
        """Fit the AugBoost model.

            Parameters
            ----------
            X : array-like, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number of samples
                and n_features is the number of features.

            y : array-like, shape = [n_samples]
                Target values (integers in classification, real numbers in
                regression)
                For classification, labels must correspond to classes.

            X_val : array-like, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number of samples
                and n_features is the number of features.

            y_val : array-like, shape = [n_samples]
                Target values (strings or integers in classification, real numbers
                in regression)
                For classification, labels must correspond to classes.

            sample_weight : array-like, shape = [n_samples] or None
                Sample weights. If None, then samples are equally weighted. Splits
                that would create child nodes with net zero or negative weight are
                ignored while searching for a split in each node. In the case of
                classification, splits are also ignored if they would result in any
                single class carrying a negative weight in either child node.

            monitor : callable, optional
                The monitor is called after each iteration with the current
                iteration, a reference to the estimator and the local variables of
                ``_fit_stages`` as keyword arguments ``callable(i, self,
                locals())``. If the callable returns ``True`` the fitting procedure
                is stopped. The monitor can be used for various things such as
                computing held-out estimates, early stopping, model introspect, and
                snapshoting.

        Returns
        -------
        self : object
            Returns self.
        """

        # if not warmstart - clear the estimator state
        if not self.warm_start:
            self._clear_state()
        # Check input
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'], dtype=DTYPE)
        n_samples, self.n_features_ = np.concatenate([X, X], axis=1).shape
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float32)
        else:
            sample_weight = column_or_1d(sample_weight, warn=True)

        check_consistent_length(X, y, sample_weight)
        y = self._validate_y(y, sample_weight)

        #        if self.n_iter_no_change is not None:
        #            X, X_val, y, y_val, sample_weight, sample_weight_val = (
        #                train_test_split(X, y, sample_weight,
        #                                 random_state=self.random_state,
        #                                 test_size=self.validation_fraction))
        #        else:
        #            X_val = y_val = sample_weight_val = None

        sample_weight_val = None

        self._check_params()

        if not self._is_initialized():
            # init state
            self._init_state()

            # fit initial model - FIXME make sample_weight optional
            self.init_.fit(X, y, sample_weight)

            # init predictions
            y_pred = self.init_.predict(X)
            begin_at_stage = 0

            # The rng state must be preserved if warm_start is True
            self._rng = check_random_state(self.random_state)

        else:
            # add more estimators to fitted model
            # invariant: warm_start = True
            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError('n_estimators=%d must be larger or equal to '
                                 'estimators_.shape[0]=%d when '
                                 'warm_start==True'
                                 % (self.n_estimators,
                                    self.estimators_.shape[0]))
            begin_at_stage = self.estimators_.shape[0]
            # The requirements of _decision_function (called in two lines
            # below) are more constrained than fit. It accepts only CSR
            # matrices.
            X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
            y_pred = self._decision_function(X)
            self._resize_state()

        if self.presort is True and issparse(X):
            raise ValueError(
                "Presorting is not supported for sparse matrices.")

        presort = self.presort
        # Allow presort to be 'auto', which means True if the dataset is dense,
        # otherwise it will be False.
        if presort == 'auto':
            presort = not issparse(X)

        X_idx_sorted = None
        if presort:
            X_idx_sorted = np.asfortranarray(np.argsort(np.concatenate([X, X], axis=1), axis=0),
                                             dtype=np.int32)

        # fit the boosting stages
        n_stages = self._fit_stages(X, y, y_pred, sample_weight, self._rng,
                                    X_val, y_val, sample_weight_val,
                                    begin_at_stage, monitor, X_idx_sorted)

        # change shape of arrays after fit (early-stopping or additional ests)
        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            self.val_score_ = self.val_score_[:n_stages]
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = self.oob_improvement_[:n_stages]

        self.n_estimators_ = n_stages
        return self

    def _fit_stages(self, X, y, y_pred, sample_weight, random_state,
                    X_val, y_val, sample_weight_val,
                    begin_at_stage=0, monitor=None, X_idx_sorted=None):
        """Iteratively fits the stages.

        For each stage it computes the progress (OOB, train score)
        and delegates to ``_fit_stage``.
        Returns the number of stages fit; might differ from ``n_estimators``
        due to early stopping.
        """

        n_samples = X.shape[0]
        do_oob = self.subsample < 1.0
        sample_mask = np.ones((n_samples,), dtype=np.bool)
        n_inbag = max(1, int(self.subsample * n_samples))
        loss_ = self.loss_

        # Set min_weight_leaf from min_weight_fraction_leaf
        if self.min_weight_fraction_leaf != 0. and sample_weight is not None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))
        else:
            min_weight_leaf = 0.

        if self.verbose:
            verbose_reporter = VerboseReporter(self.verbose)
            verbose_reporter.init(self, begin_at_stage)

        X_csc = csc_matrix(X) if issparse(X) else None
        X_csr = csr_matrix(X) if issparse(X) else None

        if self.n_iter_no_change is not None:
            loss_history = np.ones(self.n_iter_no_change) * np.inf
            # We create a generator to get the predictions for X_val after
            # the addition of each successive stage
            y_val_pred_iter = self._staged_decision_function(X_val)

        # perform boosting iterations
        i = begin_at_stage

        X_original = X
        X_normed = self.normalizer.fit_transform(X)

        for i in range(begin_at_stage, self.n_estimators):

            # subsampling
            if do_oob:
                sample_mask = _random_sample_mask(n_samples, n_inbag,
                                                  random_state)
                # OOB score before adding this stage
                old_oob_score = loss_(y[~sample_mask],
                                      y_pred[~sample_mask],
                                      sample_weight[~sample_mask])

            # Change X_original to the X you want - over here
            print('training estimator #' + str(i))
            loss = self.loss_
            residuals = loss.negative_gradient(y,
                                               y_pred)  # this line is good for regression! hasn't been checked for classification, it may work anyway
            if ((i % self.trees_between_feature_update) == 0):
                self.augmentations_.append(
                    get_transformed_params(X_normed, residuals, n_features_per_subset=self.n_features_per_subset,
                                           max_epochs=self.max_epochs, \
                                           random_state=i, augmentation_method=self.augmentation_method))
            else:
                self.augmentations_.append(self.augmentations_[-1])
            X = get_transformed_matrix(X_normed, self.augmentations_[i], augmentation_method=self.augmentation_method)

            # fit next stage of trees
            y_pred = self._fit_stage(i, np.concatenate([X_original, X], axis=1), y, y_pred, sample_weight,
                                     sample_mask, random_state, X_idx_sorted,
                                     X_csc, X_csr)

            if (self.save_mid_experiment_accuracy_results):
                if self.is_classification:
                    y_val_preds = self.predict_proba(X_val)
                    y_train_preds = self.predict_proba(X_original)
                    self.val_score_[i] = log_loss(np.array(pd.get_dummies(y_val)), y_val_preds)
                    self.train_score_[i] = log_loss(pd.get_dummies(y), y_train_preds)
                else:
                    y_val_preds = self.predict(X_val)
                    y_train_preds = self.predict(X_original)
                    self.val_score_[i] = loss_(y_val, y_val_preds, sample_weight_val)
                    self.train_score_[i] = loss_(y, y_train_preds, sample_weight)

            if self.verbose > 0:
                verbose_reporter.update(i, self)

            if monitor is not None:
                early_stopping = monitor(i, self, locals())
                if early_stopping:
                    break

            # We also provide an early stopping based on the score from
            # validation set (X_val, y_val), if n_iter_no_change is set
            if self.n_iter_no_change is not None:
                # By calling next(y_val_pred_iter), we get the predictions
                # for X_val after the addition of the current stage
                validation_loss = loss_(y_val, next(y_val_pred_iter),
                                        sample_weight_val)

                # Require validation_score to be better (less) than at least
                # one of the last n_iter_no_change evaluations
                if np.any(validation_loss + self.tol < loss_history):
                    loss_history[i % len(loss_history)] = validation_loss
                else:
                    break
        if (self.save_mid_experiment_accuracy_results):
            self._save_results_and_figure_to_file()

        return i + 1

    def _save_results_and_figure_to_file(self):
        # saving sequence of losses to pickle, and saving this also plotted as a figure
        mid_process_results = {'train_score': self.train_score_[-1], 'val_score': self.val_score_[-1], \
                               'train_scores_sequence': self.train_score_, 'val_scores_sequence': self.val_score_, \
                               'experiment_time': str(datetime.datetime.now())[:19]}

        filename_for_writing = str(self.experiment_name) + '_' + str(self.max_epochs) + '_max_epochs_' + str(
            self.n_estimators) + '_trees_' + str(self.n_features_per_subset) + '_features_per_subset ' + \
                               mid_process_results['experiment_time']

        with open('results/' + filename_for_writing + '.pkl', 'wb') as f:
            pickle.dump(mid_process_results, f)

        plt.plot(mid_process_results['train_scores_sequence'])
        plt.plot(mid_process_results['val_scores_sequence'])
        plt.xlabel('# of tree in sequence')
        plt.ylabel('loss')
        plt.title('Train score: ' + str(mid_process_results['train_score'])[:5] + ', Val score: ' + str(
            mid_process_results['val_score'])[:5])
        plt.savefig('graphs/' + filename_for_writing + '.jpg')
        plt.close()

    def _decision_function(self, X):
        # for use in inner loop, not raveling the output in single-class case,
        # not doing input validation.
        score = self._predict_stages(X)
        return score

    def _staged_decision_function(self, X):
        """Compute decision function of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : generator of array, shape = [n_samples, k]
            The decision function of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
        score = self._init_decision_function(X)
        X_original = X
        X_normed = self.normalizer.transform(X)

        for i in range(self.estimators_.shape[0]):
            X = get_transformed_matrix(X_normed, self.augmentations_[i], augmentation_method=self.augmentation_method)
            predict_stage(self.estimators_, i, np.concatenate([X_original, X], axis=1), self.learning_rate, score)
            yield score.copy()

    def _validate_y(self, y, sample_weight):
        # 'sample_weight' is not utilised but is used for
        # consistency with similar method _validate_y of GBC
        self.n_classes_ = 1
        if y.dtype.kind == 'O':
            y = y.astype(np.float64)
            # Default implementation
        return y

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array_like, shape = [n_samples, n_estimators, n_classes]
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
            In the case of binary classification n_classes is 1.
        """

        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)

        # n_classes will be equal to 1 in the binary classification or the
        # regression case.
        n_estimators, n_classes = self.estimators_.shape
        leaves = np.zeros((X.shape[0], n_estimators, n_classes))

        X_original = X
        X_normed = self.normalizer.transform(X)

        for i in range(n_estimators):
            X = get_transformed_matrix(X_normed, self.augmentations_[i], augmentation_method=self.augmentation_method)
            for j in range(n_classes):
                estimator = self.estimators_[i, j]
                leaves[:, i, j] = estimator.apply(np.concatenate([X_original, X], axis=1), check_input=False)

        return leaves


class AugBoostClassifier(AugBoostBase):
    """AugBoost for classification. AugBoost builds an additive model in a forward stage-wise fashion, while augmenting the features in between "boosts" using neural networks, PCA or random projections.

       Parameters
       ----------
       loss : {'deviance', 'exponential'}, optional (default='deviance')
           loss function to be optimized. 'deviance' refers to
           deviance (= logistic regression) for classification
           with probabilistic outputs. For loss 'exponential'
           this recovers the AdaBoost algorithm.

       learning_rate : float, optional (default=0.1)
           learning rate shrinks the contribution of each tree by `learning_rate`.
           There is a trade-off between learning_rate and n_estimators.

       n_estimators : int (default=100)
           The number of boosting stages to perform.

       max_depth : integer, optional (default=3)
           maximum depth of the individual regression estimators. The maximum
           depth limits the number of nodes in the tree. Tune this parameter
           for best performance; the best value depends on the interaction
           of the input variables.

       criterion : string, optional (default="friedman_mse")
           The function to measure the quality of a split. Supported criteria
           are "friedman_mse" for the mean squared error with improvement
           score by Friedman, "mse" for mean squared error, and "mae" for
           the mean absolute error. The default value of "friedman_mse" is
           generally the best as it can provide a better approximation in
           some cases.

           .. versionadded:: 0.18

       min_samples_split : int, float, optional (default=2)
           The minimum number of samples required to split an internal node:

           - If int, then consider `min_samples_split` as the minimum number.
           - If float, then `min_samples_split` is a fraction and
             `ceil(min_samples_split * n_samples)` are the minimum
             number of samples for each split.

           .. versionchanged:: 0.18
              Added float values for fractions.

       min_samples_leaf : int, float, optional (default=1)
           The minimum number of samples required to be at a leaf node:

           - If int, then consider `min_samples_leaf` as the minimum number.
           - If float, then `min_samples_leaf` is a fraction and
             `ceil(min_samples_leaf * n_samples)` are the minimum
             number of samples for each node.

           .. versionchanged:: 0.18
              Added float values for fractions.

       min_weight_fraction_leaf : float, optional (default=0.)
           The minimum weighted fraction of the sum total of weights (of all
           the input samples) required to be at a leaf node. Samples have
           equal weight when sample_weight is not provided.

       subsample : float, optional (default=1.0)
           The fraction of samples to be used for fitting the individual base
           learners. If smaller than 1.0 this results in Stochastic Boosting.
           `subsample` interacts with the parameter `n_estimators`.
           Choosing `subsample < 1.0` leads to a reduction of variance
           and an increase in bias.

       max_features : int, float, string or None, optional (default=None)
           The number of features to consider when looking for the best split:

           - If int, then consider `max_features` features at each split.
           - If float, then `max_features` is a fraction and
             `int(max_features * n_features)` features are considered at each
             split.
           - If "auto", then `max_features=sqrt(n_features)`.
           - If "sqrt", then `max_features=sqrt(n_features)`.
           - If "log2", then `max_features=log2(n_features)`.
           - If None, then `max_features=n_features`.

           Choosing `max_features < n_features` leads to a reduction of variance
           and an increase in bias.

           Note: the search for a split does not stop until at least one
           valid partition of the node samples is found, even if it requires to
           effectively inspect more than ``max_features`` features.

       max_leaf_nodes : int or None, optional (default=None)
           Grow trees with ``max_leaf_nodes`` in best-first fashion.
           Best nodes are defined as relative reduction in impurity.
           If None then unlimited number of leaf nodes.

       min_impurity_split : float,
           Threshold for early stopping in tree growth. A node will split
           if its impurity is above the threshold, otherwise it is a leaf.

           .. deprecated:: 0.19
              ``min_impurity_split`` has been deprecated in favor of
              ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
              Use ``min_impurity_decrease`` instead.

       min_impurity_decrease : float, optional (default=0.)
           A node will be split if this split induces a decrease of the impurity
           greater than or equal to this value.

           The weighted impurity decrease equation is the following::

               N_t / N * (impurity - N_t_R / N_t * right_impurity
                                   - N_t_L / N_t * left_impurity)

           where ``N`` is the total number of samples, ``N_t`` is the number of
           samples at the current node, ``N_t_L`` is the number of samples in the
           left child, and ``N_t_R`` is the number of samples in the right child.

           ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
           if ``sample_weight`` is passed.

           .. versionadded:: 0.19

       init : estimator, optional
           An estimator object that is used to compute the initial
           predictions. ``init`` has to provide ``fit`` and ``predict``.
           If None it uses ``loss.init_estimator``.

       verbose : int, default: 0
           Enable verbose output. If 1 then it prints progress and performance
           once in a while (the more trees the lower the frequency). If greater
           than 1 then it prints progress and performance for every tree.

       warm_start : bool, default: False
           When set to ``True``, reuse the solution of the previous call to fit
           and add more estimators to the ensemble, otherwise, just erase the
           previous solution. See :term:`the Glossary <warm_start>`.

       max_epochs : int (default=10)
           Amount of maximum epochs the neural network meant for embedding will use each time
           (if early stopping isn't activated). Notice that a neural network will be trained a number of times,
           so be cautious with this parameter.

       save_mid_experiment_accuracy_results : bool (default=False)
           Whether to save the results from mid-experiment to file. The raw results will be saved as a pickle, and the graph will be saved as an image.
           If this is set as True, make sure you have a 'results' folder and a 'graphs' folder in the path from which you'll be running your experiment.
           This increases runtime significantly, therefore default is False.

       experiment_name : string (default='')
           Name of the experiment to be saved to file (details of the other parameters and timestamp will be included in the filename).
           The name should contain the dataset being tested, and also the unique charactersitics of the experiment.

       augmentation_method : string (default='nn')
           'nn', 'rp' or 'pca'. This augmentation will be used every 'trees_between_feature_update' amount of trees, to obtain new features from X.
           These features will be concatenated will the original features, and used for the next 'trees_between_feature_update' amount of trees.
           After that these featured will be "dumped" and new features will be created with the same method.

       trees_between_feature_update : int (default=10)
           This defines the frequency in which the features obtained by augmentation will be dumped, and new ones will be created.
           Notice that this parameter effects the run-time dramatically (both train and test).

       random_state : int, RandomState instance or None, optional (default=None)
           If int, random_state is the seed used by the random number generator;
           If RandomState instance, random_state is the random number generator;
           If None, the random number generator is the RandomState instance used
           by `np.random`.

       n_features_per_subset: int (default=4)
           number of features in each feature subset of the augmentation matrix.
           This is identical to the rotation for each tree in the rotation forest algorithm.

       presort : bool or 'auto', optional (default='auto')
           Whether to presort the data to speed up the finding of best splits in
           fitting. Auto mode by default will use presorting on dense data and
           default to normal sorting on sparse data. Setting presort to true on
           sparse data will raise an error.

           .. versionadded:: 0.17
              *presort* parameter.

       validation_fraction : float, optional, default 0.1
           The proportion of training data to set aside as validation set for
           early stopping. Must be between 0 and 1.
           Only used if ``n_iter_no_change`` is set to an integer.

           .. versionadded:: 0.20

       n_iter_no_change : int, default None
           ``n_iter_no_change`` is used to decide if early stopping will be used
           to terminate training when validation score is not improving. By
           default it is set to None to disable early stopping. If set to a
           number, it will set aside ``validation_fraction`` size of the training
           data as validation and terminate training when validation score is not
           improving in all of the previous ``n_iter_no_change`` numbers of
           iterations.

           .. versionadded:: 0.20

       tol : float, optional, default 1e-4
           Tolerance for the early stopping. When the loss is not improving
           by at least tol for ``n_iter_no_change`` iterations (if set to a
           number), the training stops.

           .. versionadded:: 0.20

       Attributes
       ----------
       n_estimators_ : int
           The number of estimators as selected by early stopping (if
           ``n_iter_no_change`` is specified). Otherwise it is set to
           ``n_estimators``.

           .. versionadded:: 0.20

       feature_importances_ : array, shape = [n_features]
           The feature importances (the higher, the more important the feature).

       oob_improvement_ : array, shape = [n_estimators]
           The improvement in loss (= deviance) on the out-of-bag samples
           relative to the previous iteration.
           ``oob_improvement_[0]`` is the improvement in
           loss of the first stage over the ``init`` estimator.

       train_score_ : array, shape = [n_estimators]
           The i-th score ``train_score_[i]`` is the deviance (= loss) of the
           model at iteration ``i`` on the in-bag sample.
           If ``subsample == 1`` this is the deviance on the training data.

       val_score_ : array, shape = [n_estimators]
           The i-th score ``val_score_[i]`` is the deviance (= loss) of the
           model at iteration ``i`` on the validation data.
           If ``subsample == 1`` this is the deviance on the validation data.

       nn_histories : list, shape not well defined
           Variable for saving the training history of all the NN's that were trained as part of the NetBoost model.

       loss_ : LossFunction
           The concrete ``LossFunction`` object.

       init_ : estimator
           The estimator that provides the initial predictions.
           Set via the ``init`` argument or ``loss.init_estimator``.

       estimators_ : ndarray of DecisionTreeRegressor, shape = [n_estimators, ``loss_.K``]
           The collection of fitted sub-estimators. ``loss_.K`` is 1 for binary
           classification, otherwise n_classes.


    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.
    ***This code is heavily based on scikit-learn's code repository (ensemble.gradient_boosting.py).
    This began as a fork of their code, and was modified to be an independent repository once we realized that it would be more convenient to use this way. Thanks!***
    """

    def __init__(self, loss='deviance', learning_rate=0.1, subsample=1.0,
                 n_estimators=100, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None,
                 random_state=None, max_epochs=10, n_features_per_subset=4, max_features=None,
                 augmentation_method='nn', trees_between_feature_update=10,
                 max_leaf_nodes=None, warm_start=False,
                 presort='auto', validation_fraction=0.1,
                 n_iter_no_change=None, experiment_name='AugBoost_experiment',
                 save_mid_experiment_accuracy_results=False, tol=1e-4):

        params = {'loss': loss, 'learning_rate': learning_rate, 'n_estimators': n_estimators,
                  'criterion': criterion, 'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'min_weight_fraction_leaf': min_weight_fraction_leaf,
                  'max_depth': max_depth, 'init': init, 'subsample': subsample,
                  'max_features': max_features,
                  'random_state': random_state, 'max_epochs': max_epochs,
                  'n_features_per_subset': n_features_per_subset,
                  'max_leaf_nodes': max_leaf_nodes,
                  'min_impurity_decrease': min_impurity_decrease,
                  'augmentation_method': augmentation_method,
                  'trees_between_feature_update': trees_between_feature_update,
                  'min_impurity_split': min_impurity_split,
                  'warm_start': warm_start, 'presort': presort,
                  'validation_fraction': validation_fraction,
                  'n_iter_no_change': n_iter_no_change,
                  'experiment_name': experiment_name,
                  'save_mid_experiment_accuracy_results': save_mid_experiment_accuracy_results, 'tol': tol,
                  'is_classification': True}

        params_gbdt = {
                  'loss': loss, 'learning_rate': learning_rate, 'n_estimators': n_estimators,
                  'criterion': criterion, 'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'min_weight_fraction_leaf': min_weight_fraction_leaf,
                  'max_depth': max_depth, 'init': init, 'subsample': subsample,
                  'max_features': max_features,
                  'random_state': random_state,
                  'max_leaf_nodes': max_leaf_nodes,
                  'min_impurity_decrease': min_impurity_decrease,
                  'min_impurity_split': min_impurity_split,
                  'warm_start': warm_start, 'presort': presort}

        super(AugBoostClassifier, self).__init__(**params)
        self.classifier = GradientBoostingClassifier(**params_gbdt)

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None, monitor=None):
        super(AugBoostClassifier,self).fit(X, y, X_val, y_val, sample_weight, monitor)
        self.classifier.fit(X, y, sample_weight, monitor)

    def _validate_y(self, y, sample_weight):
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return y

    def decision_function(self, X):
        return self.classifier.decision_function(self, X)

    def staged_decision_function(self, X):
        return self.classifier.staged_decision_function(self, X)

    def predict(self, X):
        return self.classifier.predict(self, X)

    def staged_predict(self, X):
        return self.classifier.staged_predict(self, X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def predict_log_proba(self, X):
        return self.classifier.predict_log_proba(self, X)

    def staged_predict_proba(self, X):
        return self.classifier.staged_predict_proba(self, X)


class AugBoostRegressor(AugBoostBase):
    """AugBoost for regression. AugBoost builds an additive model in a forward stage-wise fashion, while augmenting the features in between "boosts" using neural networks, PCA or random projections

     Parameters
     ----------
     loss : {'ls', 'lad', 'huber', 'quantile'}, optional (default='ls')
         loss function to be optimized. 'ls' refers to least squares
         regression. 'lad' (least absolute deviation) is a highly robust
         loss function solely based on order information of the input
         variables. 'huber' is a combination of the two. 'quantile'
         allows quantile regression (use `alpha` to specify the quantile).

     learning_rate : float, optional (default=0.1)
         learning rate shrinks the contribution of each tree by `learning_rate`.
         There is a trade-off between learning_rate and n_estimators.

     n_estimators : int (default=100)
         The number of boosting stages to perform.

     max_depth : integer, optional (default=3)
         maximum depth of the individual regression estimators. The maximum
         depth limits the number of nodes in the tree. Tune this parameter
         for best performance; the best value depends on the interaction
         of the input variables.

     criterion : string, optional (default="friedman_mse")
         The function to measure the quality of a split. Supported criteria
         are "friedman_mse" for the mean squared error with improvement
         score by Friedman, "mse" for mean squared error, and "mae" for
         the mean absolute error. The default value of "friedman_mse" is
         generally the best as it can provide a better approximation in
         some cases.

         .. versionadded:: 0.18

     min_samples_split : int, float, optional (default=2)
         The minimum number of samples required to split an internal node:

         - If int, then consider `min_samples_split` as the minimum number.
         - If float, then `min_samples_split` is a fraction and
           `ceil(min_samples_split * n_samples)` are the minimum
           number of samples for each split.

         .. versionchanged:: 0.18
            Added float values for fractions.

     min_samples_leaf : int, float, optional (default=1)
         The minimum number of samples required to be at a leaf node:

         - If int, then consider `min_samples_leaf` as the minimum number.
         - If float, then `min_samples_leaf` is a fraction and
           `ceil(min_samples_leaf * n_samples)` are the minimum
           number of samples for each node.

         .. versionchanged:: 0.18
            Added float values for fractions.

     min_weight_fraction_leaf : float, optional (default=0.)
         The minimum weighted fraction of the sum total of weights (of all
         the input samples) required to be at a leaf node. Samples have
         equal weight when sample_weight is not provided.

     subsample : float, optional (default=1.0)
         The fraction of samples to be used for fitting the individual base
         learners. If smaller than 1.0 this results in Stochastic
         Boosting. `subsample` interacts with the parameter `n_estimators`.
         Choosing `subsample < 1.0` leads to a reduction of variance
         and an increase in bias.

     max_features : int, float, string or None, optional (default=None)
         The number of features to consider when looking for the best split:

         - If int, then consider `max_features` features at each split.
         - If float, then `max_features` is a fraction and
           `int(max_features * n_features)` features are considered at each
           split.
         - If "auto", then `max_features=n_features`.
         - If "sqrt", then `max_features=sqrt(n_features)`.
         - If "log2", then `max_features=log2(n_features)`.
         - If None, then `max_features=n_features`.

         Choosing `max_features < n_features` leads to a reduction of variance
         and an increase in bias.

         Note: the search for a split does not stop until at least one
         valid partition of the node samples is found, even if it requires to
         effectively inspect more than ``max_features`` features.

     max_leaf_nodes : int or None, optional (default=None)
         Grow trees with ``max_leaf_nodes`` in best-first fashion.
         Best nodes are defined as relative reduction in impurity.
         If None then unlimited number of leaf nodes.

     min_impurity_split : float,
         Threshold for early stopping in tree growth. A node will split
         if its impurity is above the threshold, otherwise it is a leaf.

         .. deprecated:: 0.19
            ``min_impurity_split`` has been deprecated in favor of
            ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
            Use ``min_impurity_decrease`` instead.

     min_impurity_decrease : float, optional (default=0.)
         A node will be split if this split induces a decrease of the impurity
         greater than or equal to this value.

         The weighted impurity decrease equation is the following::

             N_t / N * (impurity - N_t_R / N_t * right_impurity
                                 - N_t_L / N_t * left_impurity)

         where ``N`` is the total number of samples, ``N_t`` is the number of
         samples at the current node, ``N_t_L`` is the number of samples in the
         left child, and ``N_t_R`` is the number of samples in the right child.

         ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
         if ``sample_weight`` is passed.

         .. versionadded:: 0.19

       max_epochs : int (default=10)
           Amount of maximum epochs the neural network meant for embedding will use each time
           (if early stopping isn't activated). Notice that a neural network will be trained a number of times,
           so be cautious with this parameter.

     save_experiment_results : bool (default=True)
         Whether to save the results from mid-experiment to file. The raw results will be saved as a pickle,
         and the graph will be saved as an image. If this is set as True, make sure you have a 'results' folder and a
         'graphs' folder in the path from which you'll be running your experiment.

     experiment_name : string (default='')
         Name of the experiment to be saved to file (details of the other parameters and timestamp will be included in
         the filename). The name should contain the dataset being tested, and also the unique charactersitics of the
         experiment. Used for documenting saved file. The current date and time will be added to this.

     augmentation_method : string (default='nn')
         'nn', 'rp' or 'pca'. This augmentation will be used every 'trees_between_feature_update' amount of trees,
         to obtain new features from X. These features will be concatenated will the original features, and used for the
         next 'trees_between_feature_update' amount of trees. After that these featured will be "dumped" and new
         features will be created with the same method.

     trees_between_feature_update : int (default=10)
         This defines the frequency in which the features obtained by augmentation will be dumped, and new ones will be
         created. Notice that this parameter effects the run-time dramatically (both train and test).

     alpha : float (default=0.9)
         The alpha-quantile of the huber loss function and the quantile
         loss function. Only if ``loss='huber'`` or ``loss='quantile'``.

     init : estimator, optional (default=None)
         An estimator object that is used to compute the initial
         predictions. ``init`` has to provide ``fit`` and ``predict``.
         If None it uses ``loss.init_estimator``.

     verbose : int, default: 0
         Enable verbose output. If 1 then it prints progress and performance
         once in a while (the more trees the lower the frequency). If greater
         than 1 then it prints progress and performance for every tree.

     warm_start : bool, default: False
         When set to ``True``, reuse the solution of the previous call to fit
         and add more estimators to the ensemble, otherwise, just erase the
         previous solution. See :term:`the Glossary <warm_start>`.

     random_state : int, RandomState instance or None, optional (default=None)
         If int, random_state is the seed used by the random number generator;
         If RandomState instance, random_state is the random number generator;
         If None, the random number generator is the RandomState instance used
         by `np.random`.

     n_features_per_subset: int (default=4)
         number of features in each feature subset of the augmentation matrix.
         This is identical to the rotation for each tree in the rotation forest algorithm.

     presort : bool or 'auto', optional (default='auto')
         Whether to presort the data to speed up the finding of best splits in
         fitting. Auto mode by default will use presorting on dense data and
         default to normal sorting on sparse data. Setting presort to true on
         sparse data will raise an error.

         .. versionadded:: 0.17
            optional parameter *presort*.

     validation_fraction : float, optional, default 0.1
         The proportion of training data to set aside as validation set for
         early stopping. Must be between 0 and 1.
         Only used if early_stopping is True

         .. versionadded:: 0.20

     n_iter_no_change : int, default None
         ``n_iter_no_change`` is used to decide if early stopping will be used
         to terminate training when validation score is not improving. By
         default it is set to None to disable early stopping. If set to a
         number, it will set aside ``validation_fraction`` size of the training
         data as validation and terminate training when validation score is not
         improving in all of the previous ``n_iter_no_change`` numbers of
         iterations.

         .. versionadded:: 0.20


     tol : float, optional, default 1e-4
         Tolerance for the early stopping. When the loss is not improving
         by at least tol for ``n_iter_no_change`` iterations (if set to a
         number), the training stops.

         .. versionadded:: 0.20


     Attributes
     ----------
     feature_importances_ : array, shape = [n_features]
         The feature importances (the higher, the more important the feature).

     oob_improvement_ : array, shape = [n_estimators]
         The improvement in loss (= deviance) on the out-of-bag samples
         relative to the previous iteration.
         ``oob_improvement_[0]`` is the improvement in
         loss of the first stage over the ``init`` estimator.

     train_score_ : array, shape = [n_estimators]
         The i-th score ``train_score_[i]`` is the deviance (= loss) of the
         model at iteration ``i`` on the in-bag sample.
         If ``subsample == 1`` this is the deviance on the training data.

     val_score_ : array, shape = [n_estimators]
         The i-th score ``val_score_[i]`` is the deviance (= loss) of the
         model at iteration ``i`` on the validation data.
         If ``subsample == 1`` this is the deviance on the validation data.

     loss_ : LossFunction
         The concrete ``LossFunction`` object.

     init_ : estimator
         The estimator that provides the initial predictions.
         Set via the ``init`` argument or ``loss.init_estimator``.

     estimators_ : ndarray of DecisionTreeRegressor, shape = [n_estimators, 1]
         The collection of fitted sub-estimators.

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.
    ***This code is heavily based on scikit-learn's code repository (ensemble.gradient_boosting.py).
    This began as a fork of their code, and was modified to be an independent repository once we realized that it would be more convenient to use this way. Thanks!***
    """
    def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, max_epochs=10, init=None, random_state=None,
                 max_features=None, n_features_per_subset=4, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 augmentation_method='nn', trees_between_feature_update=10,
                 warm_start=False, presort='auto', validation_fraction=0.1,
                 n_iter_no_change=None,
                 experiment_name='AugBoost_experiment',
                 save_mid_experiment_accuracy_results=False, tol=1e-4):

        params = {loss: loss, learning_rate: learning_rate, n_estimators: n_estimators,
                  criterion: criterion, min_samples_split: min_samples_split,
                  min_samples_leaf: min_samples_leaf,
                  min_weight_fraction_leaf: min_weight_fraction_leaf,
                  max_depth: max_depth, init: init, subsample: subsample,
                  max_features: max_features,
                  min_impurity_decrease: min_impurity_decrease,
                  min_impurity_split: min_impurity_split,
                  random_state: random_state, n_features_per_subset: n_features_per_subset,
                  max_epochs: max_epochs,
                  alpha: alpha,
                  verbose: verbose,
                  augmentation_method: augmentation_method,
                  trees_between_feature_update: trees_between_feature_update,
                  max_leaf_nodes: max_leaf_nodes,
                  warm_start: warm_start, presort: presort,
                  validation_fraction: validation_fraction,
                  n_iter_no_change: n_iter_no_change,
                  experiment_name: experiment_name,
                  save_mid_experiment_accuracy_results: save_mid_experiment_accuracy_results, tol: tol,
                  is_classification: False}

        super(AugBoostRegressor, self).__init__(**params)

        self.regressor = GradientBoostingRegressor(**params)

    def predict(self, X):
        return self.regressor.predict(self, X)

    def staged_predict(self, X):
        return self.regressor.staged_predict(self, X)

    def apply(self, X):
        return self.regressor.apply(self, X)
