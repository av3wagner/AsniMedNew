import streamlit as st  
import streamlit.components.v1 as components
from AsniDef import *
stop=18
from datetime import datetime
import time
timestart = datetime.now()
stt = time.time()
now = datetime.now() # current date and time
date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas_profiling as pp
import math
import random
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set()

# preprocessing
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, log_loss 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve, ShuffleSplit
from sklearn.model_selection import cross_val_predict as cvp
from sklearn.calibration import CalibratedClassifierCV

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
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report,f1_score,confusion_matrix,precision_score,recall_score,balanced_accuracy_score

#st.write("1. Datenmanagement")
#st.write("Load data")

matrix ="data/heard.csv"
df = pd.read_csv(matrix) 

df.rename({'Y': 'target'}, axis=1, inplace=True)
df = df.fillna(0)

st.markdown("")
col1, col2, col3 = st.columns( [40, 1, 1])
with col1:  
    st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:left">{"Набор данных для анализа"}</h2>', unsafe_allow_html=True)
    st.markdown("")

st.dataframe(df)  

train, test = train_test_split(df, test_size = 0.4)

X_train = train[train.columns.difference(['target'])]
y_train = train['target']

X_test = test[test.columns.difference(['target'])]
y_test = test['target']

cv_n_split = 2
random_state = 0
test_train_split_part = 0.2

train0, test0 = X_train, X_test
target0 = y_train
train, test, target, target_test = train_test_split(train0, target0, test_size=test_train_split_part, random_state=random_state)

st.markdown("")
col1, col2, col3 = st.columns( [1, 40, 1])
with col2:  
    st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:center">{"Базовая статистическа о данных"}</h2>', unsafe_allow_html=True)
st.markdown("")

st.write(test.describe())  

#st.write("3. Modellenauswahl")
cv_train = ShuffleSplit(n_splits=cv_n_split, test_size=test_train_split_part, random_state=random_state)

metrics_all = {1 : 'r2_score', 2: 'acc', 3 : 'rmse', 4 : 're'}
metrics_now = [1, 2, 3, 4] 

# list of accuracy of all model - amount of metrics_now * 2 (train & test datasets)
num_models = 18
acc_train = []
acc_test = []
ShuffleSplit(n_splits=2, random_state=0, test_size=0.2, train_size=None)

############### Alte Modelle ###############
# GradientBoostingClassifier
mGBC=GradientBoostingClassifier()
GBC=GradientBoostingClassifier()
    
# KNeighborsClassifier
mKNC=KNeighborsClassifier()
KNC=KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

# SVC
mSVC = SVC()
SVC = SVC(kernel="rbf", C=0.025, probability=True)

# DecisionTreeClassifier
mDTC=DecisionTreeClassifier()
DTC=DecisionTreeClassifier(criterion = 'entropy', random_state = 51)

# RandomForestClassifier
mRF=RandomForestClassifier()
RF=RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)

# XGBClassifier
mXGBC=XGBClassifier()
XGBC=XGBClassifier()

# AdaBoostClassifier
mAdaBoost=AdaBoostClassifier()
AdaBoost=AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 200),
                              n_estimators=2000,
                              learning_rate=0.1,
                              algorithm='SAMME.R',
                              random_state=1,)    


# LogisticRegression LR
LogReg=LogisticRegression()
LR=LogisticRegression(random_state = 51)

############### Neue Modelle ###############
# Linear Regression
param_grid = {'C': np.linspace(.1, 1.5, 15)}
linreg = LogisticRegression()
linreg_CV = GridSearchCV(linreg, param_grid=param_grid, cv=cv_train, verbose=False)

# Linear SVR
linear_svc = LinearSVC()
linear_svc_CV = GridSearchCV(linear_svc, param_grid={}, cv=cv_train, verbose=False)

mlp = MLPClassifier()
param_grid = {'hidden_layer_sizes': [i for i in range(2,5)],
              'solver': ['sgd'],
              'learning_rate': ['adaptive'],
              'max_iter': [1000]
              }
mlp_GS = GridSearchCV(mlp, param_grid=param_grid, cv=cv_train, verbose=False)

# Decision Tree Classifier
decision_tree = DecisionTreeClassifier()
decision_tree_CV = GridSearchCV(decision_tree, param_grid={}, cv=cv_train, verbose=False)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd_CV = GridSearchCV(sgd, param_grid={}, cv=cv_train, verbose=False)

# Ridge Classifier
ridge = RidgeClassifier()
ridge_CV = GridSearchCV(estimator=ridge, param_grid={'alpha': np.linspace(.1, 1.5, 15)}, cv=cv_train, verbose=False)

# Bagging Classifier
bagging = BaggingClassifier()
bagging_CV = GridSearchCV(estimator=bagging, param_grid={}, cv=cv_train, verbose=False)

# AdaBoost Classifier
Ada_Boost = AdaBoostClassifier()
Ada_Boost_CV = GridSearchCV(estimator=Ada_Boost, param_grid={'learning_rate' : [.01, .1, .5, 1]}, cv=cv_train, verbose=False)

# LogisticRegression
logreg = LogisticRegression()
logreg_CV = GridSearchCV(estimator=logreg, param_grid={'C' : [.1, .3, .5, .7, 1]}, cv=cv_train, verbose=False)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian_CV = GridSearchCV(estimator=gaussian, param_grid={}, cv=cv_train, verbose=False)

# Perceptron
perceptron = Perceptron()
perceptron_CV = GridSearchCV(estimator=perceptron, param_grid={}, cv=cv_train, verbose=False)

classifiers = [
    linreg_CV,
    logreg_CV,
    perceptron_CV,
    linear_svc_CV,
    mlp_GS,
    decision_tree_CV,
    sgd_CV,
    ridge_CV,
    bagging_CV,
    Ada_Boost_CV,
    GBC,
    KNC,
    DTC,
    RF,
    XGBC,
    AdaBoost,
    gaussian_CV,
    SVC
  ] 

models = pd.DataFrame({'Model': [
    'Linear Regression', 
    'Logistic Regression',
    'Perceptron',
    'Linear SVC', 
    'MLPClassifier', 
    'Decision Tree Classifier 1', 
    'Stochastic Gradient Decent', 
    'RidgeClassifier', 
    'BaggingClassifier', 
    'AdaBoostClassifier 1', 
    'GradientBoostingClassifie',
    'KNeighborsClassifier',
    'DecisionTreeClassifier 2',
    'RandomForestClassifier',
    'XGBClassifier',
    'AdaBoostClassifier 2',
    'Naive Bayes',    
    'SVC' ]})

st.markdown("")
col1, col2, col3 = st.columns( [40, 1, 1])
with col1:  
    st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:left">{"Модели для Analyse данных"}</h2>', unsafe_allow_html=True)
    st.markdown("")

st.dataframe(models)

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
"""

i=-1
result_table = pd.DataFrame(columns=['classifiers', 
                                     'fpr',
                                     'tpr',
                                     'accurate',
                                     'auc'])

for classifier in classifiers:
    i=i+1
    name_model = models.iloc[i]['Model']
    st.markdown("")
    col1, col2, col3 = st.columns( [1, 40, 1])
    with col2:  
        st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:center">{"Model: " + name_model }</h2>', unsafe_allow_html=True)
        st.markdown("")    
        
    if i < stop:
        pipe = Pipeline(steps=[('classifier', classifier)])
        model = pipe.fit(X_train, y_train) 
        acc_metrics_calc(i,model,train,test,target,target_test)
        col1, col2, col3 = st.columns( [1, 40, 1])
        with col2:  
            st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:center">{"Generate 2 plots:1.The test and training learning curve 2.The training samples vs fit times curve"}</h2>', unsafe_allow_html=True)
        st.markdown("")   
        
        plot_learning_curve(model, name_model, train, target, cv=cv_train)
        
        ypred   = model.predict(X_test)
         
        if (i==2): 
            clf = CalibratedClassifierCV(perceptron_CV) 
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[::,1]
        elif (i==3):
            #svm = LinearSVC()
            clf = CalibratedClassifierCV(linear_svc_CV) #svm) 
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[::,1]
        elif (i==6):
            clf = CalibratedClassifierCV(sgd_CV)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[::,1]            
        elif (i==7):
            clf = CalibratedClassifierCV(ridge_CV)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[::,1]            
               
        else:
            y_pred = model.predict_proba(X_test)[::,1]
  
            
        fpr, tpr, _ = roc_curve(y_test, y_pred) 
        auc         = roc_auc_score(y_test, y_pred).round(3) 
        accuracy    = accuracy_score(ypred, y_test).round(2)
        F1_Score = f1_score(y_test, ypred).round(2)
        
        Precision_Score = precision_score(y_test, ypred).round(2)
        Recall_Score = recall_score(y_test, ypred).round(2)
        Balanced_Accuracy_Score = balanced_accuracy_score(y_test, ypred).round(2)
        Classification_Report = classification_report(y_test, ypred)
        Accurate_train=pipe.score(X_train, y_train).round(2)   
        Accurate_test =pipe.score(X_test,  y_test).round(2)   
        st.markdown("")
        col1, col2, col3 = st.columns( [40, 1, 1])
        with col1:  
            st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:left">{"Основные показатели классификации модели"}</h2>', unsafe_allow_html=True)
            st.markdown("")
        
        
        st.write("model score(AUC) ", "=" , auc)
        st.write("model score Accurate (train) ", "=" , Accurate_train)
        st.write("model score Accurate (test) ",  "=" , Accurate_test)
                    
        st.write("F1 Score ", "=" , F1_Score)
        st.write("Balanced Accuracy Score ", "=" , Balanced_Accuracy_Score)
        st.write("Precision Score ",         "=" , Precision_Score)
        st.write("Recall Score ",            "=" , Recall_Score,'\n')
 
        st.markdown("")
        col1, col2, col3 = st.columns( [40, 1, 1])
        with col1:  
            st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:left">{"Classification Report"}</h2>', unsafe_allow_html=True)
            st.markdown("")
        
        st.dataframe(classification_report(y_test, ypred, output_dict=True))  
        st.markdown("")
        col1, col2, col3 = st.columns( [1, 40, 1])
        with col2:  
            st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:center">{"Heatmap of Confusion Matrix"}</h2>', unsafe_allow_html=True)
            st.markdown("")
            
        cm = confusion_matrix(y_test, ypred)
        plt.title('Heatmap of Confusion Matrix', fontsize = 15)
        fig=plt.figure(figsize=(8,5))
        sns.heatmap(cm, annot = True)
        col1, col2, col3,= st.columns([1, 7, 1])
        with col2:
            st.pyplot(fig)
  
        Roc_Auc_Score = roc_auc_score(y_test, y_pred).round(2)
    
        st.markdown("")
        col1, col2, col3 = st.columns( [1, 40, 1])
        with col2:  
            st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:center">{"График: Receiver Operating Characteristic"}</h2>', unsafe_allow_html=True)
            st.markdown("")
        fig2=plt.figure(figsize=(8,5))            
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (name_model, auc))
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1-Specificity(False Positive Rate)')
        plt.ylabel('Sensitivity(True Positive Rate)')
        #plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show() 
        col1, col2, col3,= st.columns([1, 7, 1])
        with col2:
            st.pyplot(fig2)     

#st.dataframe(acc_all)
#5. Создание графика Multi-Curve
                                                                                                                                    
i=-1
plt.style.use('seaborn-v0_8-darkgrid')
figC = plt.figure(figsize=(16,10))

for classifier in classifiers:
    i=i+1
    name_model = models.iloc[i]['Model']
  
    if i < stop:
        pipe = Pipeline(steps=[('classifier', classifier)])
        model = pipe.fit(X_train, y_train) 
        ypred   = model.predict(X_test)
        if (i==2): 
            clf = CalibratedClassifierCV(perceptron_CV) 
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[::,1]
        elif (i==3):
            #svm = LinearSVC()
            clf = CalibratedClassifierCV(linear_svc_CV) #svm) 
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[::,1]
        elif (i==6):
            clf = CalibratedClassifierCV(sgd_CV)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[::,1]            
        elif (i==7):
            clf = CalibratedClassifierCV(ridge_CV)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[::,1]            
               
        else:
            y_pred = model.predict_proba(X_test)[::,1]
             
        fpr, tpr, _ = roc_curve(y_test, y_pred) 
        auc         = roc_auc_score(y_test, y_pred).round(3) 
        accuracy    = accuracy_score(ypred, y_test).round(2)
        F1_Score = f1_score(y_test, ypred).round(2)
        
        Precision_Score = precision_score(y_test, ypred).round(2)
        Recall_Score = recall_score(y_test, ypred).round(2)
        Balanced_Accuracy_Score = balanced_accuracy_score(y_test, ypred).round(2)
        Classification_Report = classification_report(y_test, ypred)
        Accurate_train=pipe.score(X_train, y_train).round(2)   
        Accurate_test =pipe.score(X_test,  y_test).round(2)        
         
        plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (name_model, auc))
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1-Specificity(False Positive Rate)')
        plt.ylabel('Sensitivity(True Positive Rate)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

st.markdown("")
col1, col2, col3 = st.columns( [1, 40, 1])
with col2:  
    st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:center">{"График Receiver Operating Characteristic для всех моделей"}</h2>', unsafe_allow_html=True)
    st.markdown("")
    
col1, col2, col3,= st.columns([1, 7, 1])
with col2:
    st.pyplot(figC)   
        
#5. Erstellung der ROC-Kurven
from datetime import datetime
timeend = datetime.now()
date_time = timeend.strftime("%m/%d/%Y, %H:%M:%S")
#print("Finish Analysen: ",date_time)
timedelta = (timeend-timestart)
#print ("Ende Analysen: " + str(timedelta) + " seconds")

#6. Speicher der Ergebnisse
metricsnow = pd.DataFrame(metrics_now)
metricsnow.to_csv('data/Nostalgi2023_metrics_now.csv', index=False) 
accall= pd.DataFrame(acc_all)
accall.to_csv('data/Nostalgi2023_acc_all.csv', index=False)   

st.markdown("")
col1, col2, col3 = st.columns( [1, 40, 1])
with col2:  
    st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:center">{"Моделирование и оценка результатов"}</h2>', unsafe_allow_html=True)
st.markdown("")
models = pd.DataFrame({'Model': [
    'Linear Regression', 
    'Logistic Regression',
    'Perceptron',
    'Linear SVC', 
    'MLPClassifier', 
    'Decision Tree Classifier 1', 
    'Stochastic Gradient Decent', 
    'RidgeClassifier', 
    'BaggingClassifier', 
    'AdaBoostClassifier 1', 
    'GradientBoostingClassifie',
    'KNeighborsClassifier',
    'DecisionTreeClassifier 2',
    'RandomForestClassifier',
    'XGBClassifier',
    'AdaBoostClassifier 2',
    'Naive Bayes',    
    'SVC' ]})

metrics_all = {1 : 'r2_score', 2: 'acc', 3 : 'rmse', 4 : 're'}
metrics_now = [1, 2, 3, 4]

models=models[0:stop] 
for x in metrics_now:
    xs = metrics_all[x]
    models[xs + '_Train'] = acc_all[(x-1)*2]
    models[xs + '_Test']  = acc_all[(x-1)*2+1]
    if xs == "acc":
        models[xs + '_Diff'] = models[xs + '_Train'] - models[xs + '_Test']

ms = metrics_all[metrics_now[1]] # the accuracy
models.sort_values(by=[(ms + '_Test'), (ms + '_Train')], ascending=False)
pd.options.display.float_format = '{:,.2f}'.format

# Plots
plt.style.use('seaborn-v0_8-darkgrid')
for x in metrics_now: 
    st.markdown("")
    col1, col2, col3 = st.columns( [1, 40, 1])
    with col2:  
        st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:center">{str(xs) + " criterion for " + str(num_models) + " popular models for train and test datasets"}</h2>', unsafe_allow_html=True)
        st.markdown("")
    xs = metrics_all[x]
    xs_train = metrics_all[x] + '_Train'
    xs_test = metrics_all[x] + '_Test'
    figD = plt.figure(figsize=[25,6])
    xx = models['Model']
    plt.tick_params(labelsize=14)
    plt.plot(xx, models[xs_train], label = xs_train)
    plt.plot(xx, models[xs_test], label = xs_test)
    plt.legend()
    plt.title(str(xs) + ' criterion for ' + str(num_models) + ' popular models for train and test datasets')
    plt.xlabel('Models')
    plt.ylabel(xs + ', %')
    plt.xticks(xx, rotation='vertical')
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot(figD)   

    
# Choose the number of metric by which the best models will be determined =>  {1 : 'r2_score', 2 : 'relative_error', 3 : 'rmse'}
metrics_main = 2
xs = metrics_all[metrics_main]
xs_train = metrics_all[metrics_main] + '_Train'
xs_test = metrics_all[metrics_main] + '_Test'
direct_sort = False if (metrics_main >= 2) else True

models_sort = models.sort_values(by=[xs_test, xs_train], ascending=direct_sort)
st.markdown("")
col1, col2, col3 = st.columns( [1, 40, 1])
with col2:  
    st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:center">{"Лучшие моделии по критериям выбора: acc_Diff < 5 and acc_Train > 90"}</h2>', unsafe_allow_html=True)
st.markdown("")

st.dataframe(models_sort)
models_best = models_sort[(models_sort.acc_Diff < 5) & (models_sort.acc_Train > 90)]

models_best[['Model', ms + '_Train', ms + '_Test', 'acc_Diff']].sort_values(by=['acc_Test'], ascending=False)
st.markdown("")
col1, col2, col3 = st.columns( [1, 40, 1])
with col2:  
    st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:center">{"Лучшие моделии отсортированные по показателю: acc_Test"}</h2>', unsafe_allow_html=True)
st.markdown("")
st.dataframe(models_best)

from datetime import datetime
timeend = datetime.now()
date_time = timeend.strftime("%m/%d/%Y, %H:%M:%S")
#st.write("Finish best models: " +str(date_time))
timedelta = (timeend-timestart)
#st.write("Ender Teil I: " + str(timedelta) + " seconds") 

st.markdown("")
col1, col2, col3 = st.columns( [1, 40, 1])
with col2:  
    st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:center">{"Рассчёт прогноза по всем моделям"}</h2>', unsafe_allow_html=True)
st.markdown("")

metrics_all = {1 : 'r2_score', 2: 'acc', 3 : 'rmse', 4 : 're'}
metrics_now = [1, 2, 3, 4] 

# list of accuracy of all model - amount of metrics_now * 2 (train & test datasets)
num_models = 18
acc_train = []
acc_test = []
acc_all = np.empty((len(metrics_now)*2, 0)).tolist()

models = pd.DataFrame({'Model': [
    'Linear Regression', 
    'Logistic Regression',
    'Perceptron',
    'Linear SVC', 
    'MLPClassifier', 
    'Decision Tree Classifier 1', 
    'Stochastic Gradient Decent', 
    'RidgeClassifier', 
    'BaggingClassifier', 
    'AdaBoostClassifier 1', 
    'GradientBoostingClassifie',
    'KNeighborsClassifier',
    'DecisionTreeClassifier 2',
    'RandomForestClassifier',
    'XGBClassifier',
    'AdaBoostClassifier 2',
    'SVC'
 ]})

#st.write(models)
models_pred = pd.DataFrame(models.Model, columns = ['Model'])
N_best_models = len(models_pred.Model)

from sklearn.svm import SVC
i=0
for i in range(N_best_models):
    if i < N_best_models+1:
        name_model = models_pred.iloc[i]['Model']
        ii=i+1
        # model from Sklearn
        model = model_fit(name_model,train0,target0)
        acc_metrics_calc_pred(i,model,name_model, train0, test0, target0)

from datetime import datetime
timeend = datetime.now()
date_time = timeend.strftime("%m/%d/%Y, %H:%M:%S")
#st.write("Finish: " + str(date_time))
timedelta = (timeend-timestart)
#st.write("Gesamtzeit für komplette Berechnung der Prediction: " + str(timedelta) + " seconds") 
elapsed_time = time.time() - stt
#st.write("Gesamtzeit für komplette Berechnung der Prediction: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

models_pred=models_pred[0:N_best_models]

i=0
for x in metrics_now:
    xs = metrics_all[x]
    #Auswahl N_best_models(17 im Moment) Spalten aus der acc_all_pred!  
    acc_all_pred2=acc_all_pred[x-1][0:N_best_models] 
    models_pred[xs + '_train'] = acc_all_pred2 

     
#st.markdown("")
#col1, col2, col3 = st.columns( [1, 40, 1])
#with col2:  
#    st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:center">{"Прогноз по всем моделям"}</h2>', unsafe_allow_html=True)
#st.markdown("")     

st.dataframe(models_pred)
sort_pred=models_pred[['Model', 'acc_train']].sort_values(by=['acc_train'], ascending=False)
sort_pred.to_csv('data/Nostalgi2023_BestModel.csv', index=False) 

st.markdown("")
col1, col2, col3 = st.columns( [1, 40, 1])
with col2:  
    st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:center">{"Mодели прогноза отсортированные по показателю: acc_train"}</h2>', unsafe_allow_html=True)
st.markdown("")

st.dataframe(sort_pred)

from datetime import datetime
timeend = datetime.now()
date_time = timeend.strftime("%m/%d/%Y, %H:%M:%S")
#st.write("Finish: " + str(date_time))
timedelta = (timeend-timestart)
#st.write("Gesamtzeit für komplette Berechnung models_pred: " + str(timedelta) + " seconds")    
elapsed_time = time.time() - stt
#st.write("Gesamtzeit für komplette Berechnung models_pred: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

models_pred=models_pred[0:N_best_models]

i=0
for x in metrics_now:
    xs = metrics_all[x]
    #Auswahl N_best_models(17 im Moment) Spalten aus der acc_all_pred!  
    acc_all_pred2=acc_all_pred[x-1][0:N_best_models] 
    models_pred[xs + '_train'] = acc_all_pred2 

st.markdown("")
col1, col2, col3 = st.columns( [1, 40, 1])
with col2:  
    st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:center">{"Mодели прогноза"}</h2>', unsafe_allow_html=True)
    st.markdown("")
st.dataframe(models_pred)

sort_pred=models_pred[['Model', 'r2_score_train','acc_train','rmse_train', 're_train']].sort_values(by=['acc_train'], ascending=False)
sort_pred.to_csv('data/Nostalgi2023_BestModel.csv', index=False) 

st.markdown("")
col1, col2, col3 = st.columns( [1, 40, 1])
with col2:  
    st.markdown(f'<h2 style="color:yellow;font-size:24px;text-align:center">{"Mодели прогноза, отсортированные по показателю acc_train (в убывающем порядке)"}</h2>', unsafe_allow_html=True)
    st.markdown("")
st.dataframe(sort_pred)

from datetime import datetime
timeend = datetime.now()
date_time = timeend.strftime("%d/%m/%Y, %H:%M:%S")
st.write("Финиш работы программы: " + str(date_time))
timedelta = timeend-timestart

elapsed_time = time.time() - stt
st.write("Общее время работы программы составляет: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
