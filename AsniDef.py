import numpy as np
import pandas as pd
import streamlit as st  
import streamlit.components.v1 as components

# models
from sklearn.linear_model import LogisticRegression, LogisticRegression, Perceptron, RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC, SVR, NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# ensemble
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier

from sklearn.metrics import roc_auc_score
from scikitplot.metrics import plot_roc_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import RocCurveDisplay, roc_curve
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn import metrics
from sklearn.metrics import f1_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve, ShuffleSplit
import matplotlib.pyplot as plt
import configparser

config = configparser.ConfigParser()
config.read('ProPath.ini')
pfad   = config.get('path', 'dirdata')

cv_n_split = 2
random_state = 0
test_train_split_part = 0.2
metrics_now = [1, 2, 3, 4] 
global acc_all
global acc_all_pred
acc_all = np.empty((len(metrics_now)*2, 0)).tolist()
acc_all_pred = np.empty((len(metrics_now), 0)).tolist()

cv_train = ShuffleSplit(n_splits=cv_n_split, test_size=test_train_split_part, random_state=random_state)

def delete_paragraph(paragraph):
    p = paragraph._element
    p.getparent().remove(p)
    p._p = p._element = None

def acc_d(y_meas, y_pred):
    # Relative error between predicted y_pred and measured y_meas values
    return mean_absolute_error(y_meas, y_pred)*len(y_meas)/sum(abs(y_meas))

def acc_rmse(y_meas, y_pred):
    # RMSE between predicted y_pred and measured y_meas values
    return (mean_squared_error(y_meas, y_pred))**0.5

def acc_metrics_calc(num,model,train,test,target,target_test):
    metrics_all = {1 : 'r2_score', 2: 'acc', 3 : 'rmse', 4 : 're'}
    metrics_now = [1, 2, 3, 4] 
    ytrain = model.predict(train).astype(int)
    ytest = model.predict(test).astype(int)
 
    num_acc = 0
    for x in metrics_now:
        if x == 1:
            #r2_score criterion
            acc_train = round(r2_score(target, ytrain) * 100, 2)
            acc_test  = round(r2_score(target_test, ytest) * 100, 2)
        elif x == 2:
            #accuracy_score criterion
            acc_train = round(metrics.accuracy_score(target, ytrain) * 100, 2)
            acc_test = round(metrics.accuracy_score(target_test, ytest) * 100, 2)
        elif x == 3:
            #rmse criterion
            acc_train = round(acc_rmse(target, ytrain) * 100, 2)
            acc_test = round(acc_rmse(target_test, ytest) * 100, 2)
        elif x == 4:
            #relative error criterion
            acc_train = round(acc_d(target, ytrain) * 100, 2)
            acc_test  = round(acc_d(target_test, ytest) * 100, 2)
        
        acc_all[num_acc].append(acc_train) #train
        acc_all[num_acc+1].append(acc_test) #test
        num_acc += 2

    accall= pd.DataFrame(acc_all)
    accall.to_csv(pfad+'/accall2023.csv', index=False)   
    return acc_all

def acc_metrics_calc_pred(num,model,name_model,train,test,target):
    ytrain = model.predict(train).astype(int)
    ytest = model.predict(test).astype(int)
    
    num_acc = 0
    for x in metrics_now:
        if x == 1:
            #r2_score criterion
            acc_train = round(r2_score(target, ytrain) * 100, 2)
        elif x == 2:
            #accuracy_score criterion
            acc_train = round(metrics.accuracy_score(target, ytrain) * 100, 2)
        elif x == 3:
            #rmse criterion
            acc_train = round(acc_rmse(target, ytrain) * 100, 2)
        elif x == 4:
            #relative error criterion
            acc_train = round(acc_d(target, ytrain) * 100, 2)

        acc_all_pred[num_acc].append(acc_train) #train
        num_acc += 1
        
    allpred= pd.DataFrame(acc_all_pred)
    allpred.to_csv(pfad+'/allpred2023.csv', index=False)   
    return acc_all_pred

def plot_learning_curve(estimator, title, X, y, cv=None, axes=None, ylim=None, 
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), random_state=0):
    """
    Generate 2 plots: 
    - the test and training learning curve, 
    - the training samples vs fit times curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    
    random_state : random_state
    
    """
    figure, axes = plt.subplots(2, 1, figsize=(20, 10))
    
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    cv_train = ShuffleSplit(n_splits=cv_n_split, test_size=test_train_split_part, random_state=random_state)
    
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator=estimator, X=X, y=y, cv=cv,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot(figure)   
    return
        
def model_fit(name_model,train,target):
    if name_model == 'LGBMClassifier':
        Xtrain, Xval, Ztrain, Zval = train_test_split(train, target, test_size=test_train_split_part, random_state=random_state)
        model = lgb.LGBMClassifier(n_estimators=1000)
        model.fit(Xtrain, Ztrain, eval_set=[(Xval, Zval)], early_stopping_rounds=50, verbose=False)
        
    else:
        param_grid={}
        
        if name_model == 'Linear Regression':
            model_clf = LogisticRegression()
            param_grid = {'C': np.linspace(.1, 1.5, 15)}        
        
        elif name_model == 'Support Vector Machines SVC':    
            model_clf = SVC()
            param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'tol': [1e-4]}
        
        elif name_model == 'Linear SVC':
            model_clf = LinearSVC()
            
        elif name_model == 'MLPClassifier':
            model_clf = MLPClassifier()
            param_grid = {'hidden_layer_sizes': [i for i in range(2,5)],
                          'solver': ['sgd'],
                          'learning_rate': ['adaptive'],
                          'max_iter': [1000]
                          }

        elif name_model == 'Stochastic Gradient Decent':
            model_clf = SGDClassifier()

        elif (name_model == 'DecisionTreeClassifier 2') or (name_model == "Decision Tree Classifier 1"):
            model_clf = DecisionTreeClassifier()

        elif name_model == 'RandomForestClassifier':
            model_clf = RandomForestClassifier()

        elif name_model == 'XGBClassifier':
            model_clf = xgb.XGBClassifier(objective='reg:squarederror') 
            param_grid = {'n_estimators': [50, 100, 150], 
                          'learning_rate': [0.001, 0.003, 0.005, 0.006, 0.01],
                          'max_depth': [4, 5, 6, 7]}
                        
        elif name_model == 'GradientBoostingClassifie':
            model_clf = GradientBoostingClassifier()

        elif name_model == 'RidgeClassifier':
            model_clf = RidgeClassifier()
            param_grid={'alpha': np.linspace(.1, 1.5, 15)}

        elif name_model == 'BaggingClassifier':
            model_clf = BaggingClassifier()

        elif name_model == 'ExtraTreesClassifier':
            model_clf = ExtraTreesClassifier()
            param_grid={'min_samples_leaf' : [10, 20, 30, 40, 50]}

        elif (name_model == 'AdaBoostClassifier 1') or (name_model == 'AdaBoostClassifier 2'):
            model_clf = AdaBoostClassifier()
            param_grid={'learning_rate' : [.01, .1, .5, 1]}

        elif name_model == 'Logistic Regression':
            model_clf = LogisticRegression()
            param_grid={'C' : [.1, .3, .5, .7, 1]}

        elif name_model == 'KNeighborsClassifier':
            model_clf = KNeighborsClassifier()
            param_grid={'n_neighbors': range(2, 7)}

        elif name_model == 'Naive Bayes':
            model_clf = GaussianNB()

        elif name_model == 'Perceptron':
            model_clf = Perceptron()
            
        elif name_model == 'Gaussian Process Classification':
            model_clf = GaussianProcessClassifier()
            
        elif name_model == 'SVC':
            model_clf = SVC()
            param_grid = {'kernel': ['rbf'], 'C': [0.025], 'tol': [1e-4]}
            
        model = GridSearchCV(model_clf, param_grid=param_grid, cv=cv_train, verbose=False)
        model.fit(train, target)
    return model

