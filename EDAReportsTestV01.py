#https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app/embed-your-app

print("#;;*****************************************************************;;;")
print("#;;*****************************************************************;;;")
print("#;;;****************************************************************;;;")
print("#;;;***  FIRMA          : PARADOX                                ***;;;")
print("#;;;***  Autor          : Alexander Wagner                       ***;;;")
print("#;;;***  STUDIEN-NAME   : AsniMed                                ***;;;")
print("#;;;***  STUDIEN-NUMMER :                                        ***;;;")
print("#;;;***  SPONSOR        :                                        ***;;;")
print("#;;;***  ARBEITSBEGIN   : 01.11.2023                             ***;;;")
print("#;;;****************************************************************;;;")
print("#;;;*--------------------------------------------------------------*;;;")
print("#;;;*---  PROGRAMM      : EDAReportsTestV01.ipynb               ---*;;;")
print("#;;;*---  Parent        : EDAmodReport2025Git.ipynb             ---*;;;")
print("#;;;*---  BESCHREIBUNG  : System                                ---*;;;")
print("#;;;*---                :                                       ---*;;;")
print("#;;;*---                :                                       ---*;;;")
print("#;;;*---  VERSION   VOM : 11.05.2025                            ---*;;;")
print("#;;;*--   KORREKTUR VOM :                                       ---*;;;")
print("#;;;*--                 :                                       ---*;;;")
print("#;;;*---  INPUT         :.INI, .Json, .CSV                      ---*;;;")
print("#;;;*---  OUTPUT        :.Jpg, .Png                             ---*;;;")
print("#;;;*--------------------------------------------------------------*;;;")
print("#;;;************************ Änderung ******************************;;;")
print("#;;;****************************************************************;;;")
print("#;;;  Wann              :               Was                        *;;;")
print("#;;;*--------------------------------------------------------------*;;;")
print("#;;;* 11.05.2025        : New-Progtam                              *;;;")
print("#;;;* 12.05.2025        : Korr: path                               *;;;")
print("#;;;****************************************************************;;;")

import os, sys, inspect, time, datetime
import pandas as pd
import numpy as np
import dash_pdf
from dash import Dash, html, dcc, Input, Output, State
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import dash
from dash import dash_table
from dash import Dash, dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
import chart_studio.plotly as py 
from jupyter_dash import JupyterDash
import flask
import json
import requests
from urllib.request import urlopen
from pandas_datareader import data, wb
import base64
import subprocess
import json
from time import time, strftime, localtime
from datetime import timedelta
import shutil
from subprocess import Popen, PIPE, STDOUT
import sys
import webbrowser
from configparser import ConfigParser
import streamlit as st
import matplotlib.pyplot as plt
from IPython.display import IFrame
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import psutil
from matplotlib import *
from matplotlib.colors import ListedColormap
import matplotlib
import plotly.express as px
from matplotlib import pyplot as plt
import seaborn as sns
import os, sys, inspect, time, datetime
from time import time, strftime, localtime
from datetime import timedelta
from pathlib import Path
import time
import plotly.figure_factory as ff
import plotly.io as pio
import plotly as pl
import plotly as pplt
import plotly.graph_objects as go
import plotly.offline
import plotly.offline as po
import cufflinks as cf
import patchworklib as pw
from plotly.subplots import make_subplots
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import xlsxwriter

cwd=os.getcwd()
os.chdir(cwd)
print(cwd)
pfad=cwd 
pathIm = cwd + '/assets/image'
print("pathIm: ", pathIm) 

now=datetime.datetime.now()
timestart = now.replace(microsecond=0)
print("Programm Start: ", timestart)

def boxplots_custom(dataset, columns_list, rows, cols, suptitle):
    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(13,5))
    fig.suptitle(suptitle,y=1, size=25)
    axs = axs.flatten()
    for i, data in enumerate(columns_list):
        sns.boxplot(data=dataset[data], orient='h', ax=axs[i])
        axs[i].set_title(data + ', skewness is: '+str(round(dataset[data].skew(axis = 0, skipna = True),2)))

def replace_zero_cholesterol(df):
    # Step 1: Create age groups and calculate average cholesterol for each group
    age_bins = [10, 20, 30, 40, 50, 60, 70, 80]
    age_labels = [f'{start}-{end}' for start, end in zip(age_bins[:-1], age_bins[1:])]
    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    average_cholesterol_by_age = df.groupby('AgeGroup')['Cholesterol'].mean()

    def replace_zero(row):
        if row['Cholesterol'] == 0:
            return average_cholesterol_by_age[row['AgeGroup']]
        else:
            return row['Cholesterol']

    df['Cholesterol'] = df.apply(replace_zero, axis=1)

    # Drop the temporary 'AgeGroup' column
    df.drop(columns=['AgeGroup'], inplace=True)

########################## Test1.py #############################
print("Start Test1!")

try:
    raw_df = pd.read_csv('data/heart.csv')
except:
    raw_df = pd.read_csv('data/heart.csv')

pio.renderers
def auto_fmt (pct_value):
    return '{:.0f}\n({:.1f}%)'.format(raw_df['HeartDisease'].value_counts().sum()*pct_value/100,pct_value) 

print(raw_df.head())  
HDValues={
    0:'Healthy',
    1:'Heart Disease'
    }

df = raw_df.HeartDisease.replace(HDValues)
df.info()
print(df)

pd.set_option("display.max_rows",None) 

des0=raw_df[raw_df['HeartDisease']==0].describe().T.applymap('{:,.2f}'.format)
des1=raw_df[raw_df['HeartDisease']==1].describe().T.applymap('{:,.2f}'.format)

cat = ['Sex', 'ChestPainType','FastingBS','RestingECG','ExerciseAngina',  'ST_Slope','HeartDisease']
num = ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']
numerical_columns = []
categorical_columns = []

numerical_columns = list(raw_df.loc[:,['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'HeartDisease']])
categorical_columns = list(raw_df.loc[:,['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']])
numerical=numerical_columns

index = 0
plt.figure(figsize=(20,20))
for feature in numerical:
    if feature != "HeartDisease":
        index += 1
        plt.subplot(2, 3, index)
        sns.boxplot(x='HeartDisease', y=feature, data=raw_df)
        
plt.savefig(pathIm + '/EDA1.png')  

print('numerical_columns before clear:', numerical_columns)
numerical_columns.clear()
print('numerical_columns after clear:', numerical_columns)
del numerical_columns[:]
del categorical_columns[:]
numerical_columns = list(raw_df.loc[:,['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'HeartDisease']])
categorical_columns = list(raw_df.loc[:,['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']])

# checking boxplots
boxplots_custom(dataset=raw_df, columns_list=numerical_columns, rows=2, cols=3, suptitle='Boxplots for each variable')
plt.tight_layout()
plt.savefig(pathIm + '/EDA2.png')

fig, axes = plt.subplots(nrows=3, ncols=2,figsize=(11,17))
fig.suptitle('Features vs Class\n', size = 18)

sns.boxplot(ax=axes[0, 0], data=raw_df, x='Sex', y='Age', palette='Spectral')
axes[0,0].set_title("Age distribution");


sns.boxplot(ax=axes[0,1], data=raw_df, x='Sex', y='RestingBP', palette='Spectral')
axes[0,1].set_title("RestingBP distribution");


sns.boxplot(ax=axes[1, 0], data=raw_df, x='Sex', y='Cholesterol', palette='Spectral')
axes[1,0].set_title("Cholesterol distribution");

sns.boxplot(ax=axes[1, 1], data=raw_df, x='Sex', y='MaxHR', palette='Spectral')
axes[1,1].set_title("MaxHR distribution");

sns.boxplot(ax=axes[2, 0], data=raw_df, x='Sex', y='Oldpeak', palette='Spectral')
axes[2,0].set_title("Oldpeak distribution");

sns.boxplot(ax=axes[2, 1], data=raw_df, x='Sex', y='HeartDisease', palette='Spectral')
axes[2,1].set_title("HeartDisease distribution");

plt.tight_layout()
plt.savefig(pathIm + '/EDA3.png')

######################## numeric_columns ################################
numeric_columns = list(raw_df.loc[:,['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'HeartDisease']])

fig = plt.figure(figsize=(15, 10))
plt.title('Kdeplot для цифровых переменных, категория: HeartDisease')
rows, cols = 2, 3
for idx, num in enumerate(numeric_columns[:30]):
    ax = fig.add_subplot(rows, cols, idx+1)
    ax.grid(alpha = 0.7, axis ="both")
    sns.kdeplot(x = num, fill = True,color ="#3386FF",linewidth=0.6, data = raw_df[raw_df['HeartDisease']==0], label = "Healthy")
    sns.kdeplot(x = num, fill = True,color ="#EFB000",linewidth=0.6, data = raw_df[raw_df['HeartDisease']==1] , label = "Heart Disease")
    ax.set_xlabel(num)
    ax.legend()
    
fig.tight_layout()
plt.savefig(pathIm + '/EDA4.png')

fig = plt.figure(figsize=(15, 10))
plt.title('Kdeplot для цифровых переменных, категория: Sex')
rows, cols = 2, 3
for idx, num in enumerate(numeric_columns[:30]):
    ax = fig.add_subplot(rows, cols, idx+1)
    ax.grid(alpha = 0.7, axis ="both")
    sns.kdeplot(x = num, fill = True,color ="#3386FF",linewidth=0.6, data = raw_df[raw_df['Sex']=="M"], label = "M")
    sns.kdeplot(x = num, fill = True,color ="#EFB000",linewidth=0.6, data = raw_df[raw_df['Sex']=="F"], label = "F")
    ax.set_xlabel(num)
    ax.legend()
    
fig.tight_layout()
plt.savefig(pathIm + '/EDA5.png')

######################### Finish.py ##############################
print("Программа стартовала: ", timestart)
timeend = datetime.datetime.now()
date_time = timeend.strftime("%d.%m.%Y %H:%M:%S")
print("Программа финишировала:",date_time)
timedelta = round((timeend-timestart).total_seconds(), 2) 

r=(timeend-timestart) 
t=int(timedelta/60)
if timedelta-t*60 < 10:
    t2=":0" + str(int(timedelta-t*60))
else:
    t2=":" + str(int(timedelta-t*60))
txt="Длительность работы программы: 00:" + str(t) + t2 
print(txt)
print("Программа RunModuleAsniMed.py работу закончила")
