# AugBoost
Gradient Boosting Enhanced with Step-Wise Feature Augmentation. 
## About
The code in this repository is based heavily on scikit-learn's 'gradient_boosting.py'. 
We started this as a fork of sklearn, but split away when we saw it would be more convenient. Thanks! =]

## Prerequisites
* Python 3.6.5 (Ubuntu)
* sklearn.__version__ = '0.19.1'
* keras.__version__ = '2.2.0'
* tensorflow.__version__ = '1.8.0'

And a number of small packages which are included in Anaconda.
The most important prerequisite is probably the version of sklearn, although we haven't checked if any of them are necessary.

## Getting Started
After cloning the repository, the 2 modules in can be imported using these lines of code:
```python
from AugBoost import AugBoostClassifier as ABC
from AugBoost import AugBoostRegressor as ABR
```
Meanwhile, only the code for classification tasks works =[

Create your model using code that looks like this:

```python
model = ABC(n_estimators=10, max_epochs=1000, learning_rate=0.1, \
    n_features_per_subset=round(len(X_train.columns)/3), trees_between_feature_update=10,\
    augmentation_method='nn', save_mid_experiment_accuracy_results=False)
```
And then train and predict like this:
```python
model.fit(X=X_train, y=y_train)
model.predict(X_val)
```

In the file 'notebook for experiments.ipynb' there is example of code for running experiments with AugBoost.

