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

#import sqlite3
#import dash_html_components as html
#from dash.dependencies import Input, Output
#from dash import Dash, dcc, html, callback, Input, Output
#from dash import Input, Output, dcc, html
#from prophet import Prophet
#import os, sys, inspect, time, datetime

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




"""
import os, sys, inspect, time, datetime
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
import pandas as pd
import numpy as np
import sqlite3
import dash
from dash import dash_table
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash import Dash, dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
import chart_studio.plotly as py 
from jupyter_dash import JupyterDash
import flask
import json
import requests
from urllib.request import urlopen
from prophet import Prophet
from pandas_datareader import data, wb
import base64

import os, sys, inspect, time, datetime
import subprocess
import json
from time import time, strftime, localtime
from datetime import timedelta
import shutil

from subprocess import Popen, PIPE, STDOUT
import sys
import webbrowser
import pandas as pd
from configparser import ConfigParser
import streamlit as st
import matplotlib.pyplot as plt
from IPython.display import IFrame

from dash import Dash, dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import dash_ag_grid as dag
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import psutil
import dash_pdf
from dash import Dash, html, dcc, Input, Output, State

from matplotlib import *
from matplotlib.colors import ListedColormap
import matplotlib
import plotly.express as px
from matplotlib import pyplot as plt
import seaborn as sns
import os, sys, inspect, time, datetime
from time import time, strftime, localtime
from datetime import timedelta
from copy import deepcopy
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
"""

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

raw_df = pd.read_csv(pfad+'/data/heart.csv')
fig = plt.figure(figsize=(25, 10))
fig = px.scatter_3d(raw_df, 
                    x='RestingBP',
                    y='Age',
                    z='Sex',
                    color='HeartDisease')

fig.write_html(pathIm + '/Buble3D.html')

with open(pathIm + "/EDA6.png", 'wb') as f:
    f.write(pplt.io.to_image(fig, width=1200, height=800, format='png'))   

df = pd.read_csv(pfad+'/data/heart.csv')
replace_zero_cholesterol(df)

fig = px.scatter(df, y = 'Age',x='Cholesterol', color='Cholesterol' )
fig.update_layout(title=f'Buble Chart Cholesterol')
pio.write_image(fig, pathIm + '/EDA7.png', width=1200, height=800, format='png', scale=6)

cf.go_offline()
cf.set_config_file(offline=True, world_readable=True)
plt.rcParams["figure.figsize"] = (10,6)
pd.set_option('max_colwidth',200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

cat = ['Sex', 'ChestPainType','FastingBS','RestingECG',
                          'ExerciseAngina',  'ST_Slope','HeartDisease']
num = ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']

df=raw_df
fig = px.scatter(df, 
                 x=df.Age, 
                 y=df.Cholesterol, 
                 color=df.HeartDisease, 
                 facet_col=df.FastingBS,
                 facet_row=df.Sex,
                 color_discrete_map={1: "#FF5722",0: "#7CB342"},
                 width=950, 
                 height=800,
                 title="HeartDisease Data")

fig.update_layout(
                    plot_bgcolor= "#dcedc1",
                    paper_bgcolor="#FFFDE7",
                 )

fig.write_image(pathIm + '/EDA11.png',scale=4)

colors = px.colors.cyclical.Twilight

HDValues={
    0:'Healthy',
    1:'Heart Disease'
    }

df = raw_df
sns.set_theme(rc = {'figure.dpi': 250, 'axes.labelsize': 7, 
                    'axes.facecolor': '#FFFDE7', 'grid.color': '#fffdfa', 
                    'figure.facecolor': '#FFFDE7'}, font_scale = 0.55)
fig, ax = plt.subplots(4, 2, figsize = (6.5, 7.5))
for indx, (column, axes) in list(enumerate(list(zip(cat, ax.flatten())))):
    
    sns.violinplot(ax = axes, x = df[column], 
                   y = df['Age'],
                   scale = 'width', linewidth = 0.5, 
                   palette = colors, inner = None)
    
    plt.setp(axes.collections, alpha = 0.3)
    
    sns.stripplot(ax = axes, x = df[column], 
                  y = df['Age'],
                  palette = colors, alpha = 0.9, 
                  s = 1.5, jitter = 0.07)
    sns.pointplot(ax = axes, x = df[column],
                  y = df['Age'],
                  color = '#ff5736', scale = 0.25,
                  estimator = np.mean, ci = 'sd',
                  errwidth = 0.5, capsize = 0.15, join = True)
    
    plt.setp(axes.lines, zorder = 100)
    plt.setp(axes.collections, zorder = 100)
    
else:
    [axes.set_visible(False) for axes in ax.flatten()[indx + 1:]]
    
plt.tight_layout()
fig.savefig(pathIm + '/EDA12.png')


sns.set_theme(rc = {'figure.dpi': 120, 'axes.labelsize': 8, 
                    'axes.facecolor': '#FFFDE7', 'grid.color': '#fffdfa', 
                    'figure.facecolor': '#FFFDE7'}, font_scale = 0.65)

fig, ax = plt.subplots(5, 1, figsize = (10, 10))

for indx, (column, axes) in list(enumerate(list(zip(num, ax.flatten())))):
    
    sns.scatterplot(ax = axes, y = df[column].index, x = df[column], 
                    hue = df['HeartDisease'], palette = 'magma', alpha = 0.8)
    
else:
    [axes.set_visible(False) for axes in ax.flatten()[indx + 1:]]
    
plt.tight_layout()
fig.savefig(pathIm + '/EDA13.png')

sns.set_theme(rc = {'figure.dpi': 120, 'axes.labelsize': 8, 
                    'axes.facecolor': '#FFFDE7', 'grid.color': '#fffdfa', 
                    'figure.facecolor': '#FFFDE7'}, font_scale = 0.65)

fig, ax = plt.subplots(5, 1, figsize = (10, 14))

for indx, (column, axes) in list(enumerate(list(zip(num, ax.flatten())))):
    
    sns.histplot(ax = axes, x = df[column], hue = df['HeartDisease'], 
                 palette = 'magma', alpha = 0.8, multiple = 'stack')
    
    legend = axes.get_legend() # sns.hisplot has some issues with legend
    handles = legend.legendHandles
    legend.remove()
    axes.legend(handles, ['0', '1'], title = 'HeartDisease', loc = 'upper right')
    Quantiles = np.quantile(df[column], [0, 0.25, 0.50, 0.75, 1])
    
    for q in Quantiles: axes.axvline(x = q, linewidth = 0.5, color = 'r')
        
plt.tight_layout()
fig.savefig(pathIm + '/EDA14.png')

df2 = df.groupby('Sex').agg({'Age' : 'mean', "ChestPainType":'count','RestingBP':'mean','Cholesterol':'mean',
                            'FastingBS':'sum','RestingECG':'count','MaxHR':'mean','ExerciseAngina':'count','Oldpeak':'mean',
                            'ST_Slope':'count','HeartDisease':'sum'})
df2
fig=px.bar(data_frame=df2, barmode='group',
       title = "<b>Gender wise Analyzing</b>",template="plotly_dark")
fig.write_image(pathIm + '/EDA15.png',scale=4)

try:
    heart_dft = pd.read_csv(pfad+'/data/heart.csv')
except:
    heart_dft=pd.read_csv(pfad+'/data/heart.csv')

sex_color = dict({"Male": "#2986cc", "Female": "#c90076"})
plt.style.use("fivethirtyeight")
heart_dft["Sex"] = heart_dft["Sex"].map({"M": "Male", "F": "Female"})
heart_dft["Sex"]
heart_dft["HeartDisease"] = heart_dft["HeartDisease"].map({0: "No", 1: "Yes"})

filtheart_dft = heart_dft["Cholesterol"] > 0
heart_dft_chol_n0 = heart_dft[filtheart_dft]

sex_color = dict({"Male": "#2986cc", "Female": "#c90076"})
plt.style.use("fivethirtyeight")
heart_dft["Sex"] = heart_dft["Sex"].map({"M": "Male", "F": "Female"})
heart_dft["Sex"]
heart_dft["HeartDisease"] = heart_dft["HeartDisease"].map({0: "No", 1: "Yes"})
filtheart_dft = heart_dft["Cholesterol"] > 0
heart_dft_chol_n0 = heart_dft[filtheart_dft]

g=sns.JointGrid(
    data=heart_dft, x="Age", y="Cholesterol", hue="Sex", palette=sex_color
).plot(sns.scatterplot, sns.histplot)

plt.legend(title='Company', fontsize=20)
plt.xlabel('Agex', fontsize=10);
plt.ylabel('Cholesterolx', fontsize=10);
plt.title('Sales Data', fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=10)
g.savefig(pathIm + '/EDA16.png')
sns.set_theme()
pw.overwrite_axisgrid() 
iris = sns.load_dataset("iris")
tips = sns.load_dataset("tips")

# An lmplot
g0 = sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips, 
                palette=dict(Yes="g", No="m"))
g0 = pw.load_seaborngrid(g0, label="g0")

# A Pairplot
g1 = sns.pairplot(iris, hue="species")
g1 = pw.load_seaborngrid(g1, label="g1")

# A relplot
g2 = sns.relplot(data=tips, x="total_bill", y="tip", col="time", hue="time", 
                 size="size", style="sex", palette=["b", "r"], sizes=(10, 100))
g2 = pw.load_seaborngrid(g2, label="g2")

g3 = sns.jointplot(x='Cholesterol',y='Age',data=raw_df, hue="Sex") 
g3 = pw.load_seaborngrid(g3, label="g3")
(((g0|g1)["g0"]/g3)["g3"]|g2).savefig(pathIm + '/EDA17.png')

try:
    heart_dft = pd.read_csv(pfad+'/data/heart.csv')
except:
    heart_dft=pd.read_csv(pfad+'/data/heart.csv')


sex_color = dict({"Male": "#2986cc", "Female": "#c90076"})
plt.style.use("fivethirtyeight")
heart_dft["Sex"] = heart_dft["Sex"].map({"M": "Male", "F": "Female"})
heart_dft["Sex"]
heart_dft["HeartDisease"] = heart_dft["HeartDisease"].map({0: "No", 1: "Yes"})

filtheart_dft = heart_dft["Cholesterol"] > 0
heart_dft_chol_n0 = heart_dft[filtheart_dft]

Chol_mean_f = (
    heart_dft_chol_n0[["Sex", "Cholesterol"]]
    .groupby(["Sex"])
    .mean("Cholesterol")
    .loc["Female", "Cholesterol"]
).round()

Chol_mean_m = (
    heart_dft_chol_n0[["Sex", "Cholesterol"]]
    .groupby(["Sex"])
    .mean("Cholesterol")
    .loc["Male", "Cholesterol"]
).round()

plt.figure(figsize=(10, 5))
sns.set_context("paper")

kdeplt = sns.kdeplot(
    data=heart_dft_chol_n0,
    x="Cholesterol",
    hue="Sex",
    palette=sex_color,
    alpha=0.7,
    lw=2,
)

kdeplt.set_title("Cholesterol values distribution\n Male VS Female", fontsize=12)
kdeplt.set_xlabel("Cholesterol", fontsize=12)
plt.axvline(x=Chol_mean_f, color="#c90076", ls="--", lw=1.3)
plt.axvline(x=Chol_mean_m, color="#2986cc", ls="--", lw=1.3)
plt.text(108, 0.00612, "Mean Cholesterol / Male", fontsize=10, color="#2986cc")
plt.text(260, 0.006, "Mean Cholesterol / Female", fontsize=10, color="#c90076")
kdeplt.figure.savefig(pathIm + '/EDA18.png')

#######################################################
################### SAS GRAPH 19-23 ###################
#######################################################

#Kategorial
pio.renderers

try:
    raw_df = pd.read_csv(pfad+'/data/heart.csv')
except:
    raw_df = pd.read_csv(pfad+'/data/heart.csv')
HDValues={
    0:'Healthy',
    1:'Heart Disease'
    }

df = raw_df.HeartDisease.replace(HDValues)

fig=plt.figure(figsize=(6, 6))
matplotlib.rcParams.update({'font.size': 15})

df.value_counts().plot.pie(explode=[0.1, 0.1],     
                                       autopct='%1.2f%%',
                                       #autopct=auto_fmt,
                                       textprops={'fontsize': 16},
                                       shadow=True)

plt.title('Healthy vs Heart Disease', color='Red',pad=15, fontsize=20);
plt.axis('off');
plt.savefig(pathIm + '/EDA31.png')

plt.figure(figsize=(6, 6))
matplotlib.rcParams.update({'font.size': 15})

raw_df.Sex.value_counts().plot.pie(explode=[0.1, 0.1],
                                       autopct='%1.2f%%',
                                       #autopct=auto_fmt,
                                       textprops={'fontsize': 16},
                                       shadow=True)
plt.title('Sex', color='Red',pad=10, fontsize=20);
plt.axis('off');
plt.savefig(pathIm + '/EDA32.png')

fig, ax = plt.subplots (3, 2, figsize=(16, 16))
ax_rst = []
for i in range(len(categorical_columns)):
    axs = sns.countplot(data=raw_df, x =raw_df[categorical_columns[i]], ax=ax[int(i/2),i % 2])
    ax_rst.append(axs)
    total = raw_df[categorical_columns[i]].value_counts().sum()
    for p in axs.patches:
        value_pct = '{:.0f} ({:.1f}%)'.format(p.get_height(), 100 * p.get_height()/total)
        x = p.get_x() + p.get_width()/2
        y = p.get_height()
        axs.annotate(value_pct, (x, y),ha='center')   
plt.savefig(pathIm + '/EDA33.png')

fig=px.pie(raw_df,values='HeartDisease',names='ChestPainType', 
           template='plotly_dark',color_discrete_sequence=px.colors.sequential.RdBu,
           title='The effect of the type of chest pain on the disease')
fig.update_traces(textposition='inside',textinfo='percent+label')
fig.update_layout(width=1000, height=800)
fig.write_image(pathIm + '/EDA36.png', scale=4)

fig=px.pie(raw_df,values='HeartDisease',names='ST_Slope',hole=.4,template='plotly_dark',title='The effect of the the slope of the peak exercise on the disease',)
fig.update_traces(textposition='inside',textinfo='percent+label')
fig.update_layout(annotations=[dict(text='ST slope', x=0.5, y=0.5, font_size=20, showarrow=False)])
fig.update_layout(width=1000, height=1000)
fig.write_image(pathIm + '/EDA37.png',scale=4)

df=raw_df
colors = px.colors.cyclical.Twilight
fig = make_subplots(rows=1,cols=2,
                    subplot_titles=('Countplot',
                                    'Percentages'),
                    specs=[[{"type": "xy"},
                            {'type':'domain'}]])

fig.add_trace(go.Bar(y = df['Sex'].value_counts().values.tolist(), 
                      x = df['Sex'].value_counts().index, 
                      text=df['Sex'].value_counts().values.tolist(),
              textfont=dict(size=15),
                      textposition = 'outside',
                      showlegend=False,
              marker = dict(color = colors,
                            line_color = 'black',
                            line_width=3)),row = 1,col = 1)
fig.add_trace((go.Pie(labels=df['Sex'].value_counts().keys(),
                             values=df['Sex'].value_counts().values,textfont = dict(size = 16),
                     hole = .4,
                     marker=dict(colors=colors),
                     textinfo='label+percent',
                     hoverinfo='label')), row = 1, col = 2)
fig.update_yaxes(range=[0,800])
fig.update_layout(
                    paper_bgcolor= '#FFFDE7',
                    plot_bgcolor= '#FFFDE7',
                    title=dict(text = "Gender Distribution",x=0.5,y=0.95),
                    title_font_size=30
                  )
iplot(fig)
fig.write_image(pathIm + '/EDA38.png',scale=4)

colors = px.colors.cyclical.Twilight
fig = make_subplots(rows=1,cols=2,
                    subplot_titles=('Countplot',
                                    'Percentages'),
                    specs=[[{"type": "xy"},
                            {'type':'domain'}]])
fig.add_trace(go.Bar(y = df['HeartDisease'].value_counts().values.tolist(), 
                      x = df['HeartDisease'].value_counts().index, 
                      text=df['HeartDisease'].value_counts().values.tolist(),
              textfont=dict(size=15),
                      textposition = 'outside',
                      showlegend=False,
              marker = dict(color = colors,
                            line_color = 'black',
                            line_width=3)),row = 1,col = 1)
fig.add_trace((go.Pie(labels=df['HeartDisease'].value_counts().keys(),
                             values=df['HeartDisease'].value_counts().values,textfont = dict(size = 16),
                     hole = .4,
                     marker=dict(colors=colors),
                     textinfo='label+percent',
                     hoverinfo='label')), row = 1, col = 2)
fig.update_yaxes(range=[0,550])
fig.update_layout(
                    paper_bgcolor= '#FFFDE7',
                    plot_bgcolor= '#FFFDE7',
                    title=dict(text = "HeartDisease Distribution",x=0.5,y=0.95),
                    title_font_size=30
                  )
iplot(fig)
fig.write_image(pathIm + '/EDA39.png',scale=4)  

cat = ['Sex', 'ChestPainType','FastingBS','RestingECG',
                          'ExerciseAngina',  'ST_Slope','HeartDisease']
num = ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']

import seaborn as sns
sns.set_theme(rc = {'figure.dpi': 250, 'axes.labelsize': 7, 
                    'axes.facecolor': '#FFFDE7', 'grid.color': '#fffdfa', 
                    'figure.facecolor': '#FFFDE7'}, font_scale = 0.55)
fig, ax = plt.subplots(3, 2, figsize = (6.5, 9))
for indx, (column, axes) in list(enumerate(list(zip(cat, ax.flatten())))):
    if column not in 'HearDisease':
        sns.countplot(ax = axes, x = df[column], hue = df['HeartDisease'], palette = colors, alpha = 1)  
else:
    [axes.set_visible(False) for axes in ax.flatten()[indx + 1:]]   
    
axes_legend = ax.flatten()
axes_legend[1].legend(title = 'HeartDisease', loc = 'upper right')
axes_legend[2].legend(title = 'HeartDisease', loc = 'upper right')
fig.savefig(pathIm + '/EDA40.png')

sns.set_theme(rc = {'figure.dpi': 250, 'axes.labelsize': 7, 
                    'axes.facecolor': '#FFFDE7', 'grid.color': '#fffdfa', 
                    'figure.facecolor': '#FFFDE7'}, font_scale = 0.55)
fig, ax = plt.subplots(3, 2, figsize = (6.5, 9))
for indx, (column, axes) in list(enumerate(list(zip(cat[1:], ax.flatten())))):
    sns.countplot(ax = axes, x = df[column], hue = df['Sex'], palette = colors, alpha = 1)  
else:
    [axes.set_visible(False) for axes in ax.flatten()[indx + 1:]]   
axes_legend = ax.flatten()
axes_legend[1].legend(title = 'Sex', loc = 'upper right')
axes_legend[2].legend(title = 'Sex', loc = 'upper right')
fig.savefig(pathIm + '/EDA41.png')

sns.set_theme(rc = {'figure.dpi': 250, 'axes.labelsize': 7, 
                    'axes.facecolor': '#FFFDE7', 'grid.color': '#fffdfa', 
                    'figure.facecolor': '#FFFDE7'}, font_scale = 0.55)
fig, ax = plt.subplots(3, 2, figsize = (6.5, 9))
cat2 = []
for i in cat:
    if i not in 'ChestPainType':
        cat2.append(i)
for indx, (column, axes) in list(enumerate(list(zip(cat2, ax.flatten())))):
    sns.countplot(ax = axes, x = df[column], hue = df['ChestPainType'], palette = colors, alpha = 1)  
else:
    [axes.set_visible(False) for axes in ax.flatten()[indx + 1:]]   
axes_legend = ax.flatten()
axes_legend[1].legend(title = 'ChestPainType', loc = 'upper right')
axes_legend[2].legend(title = 'ChestPainType', loc = 'upper right')
fig.savefig(pathIm + '/EDA42.png')

fig, ax = plt.subplots() 
ax = plt.subplot(1,2,1)
ax = sns.countplot(x='Sex', data=raw_df)
ax.bar_label(ax.containers[0])
ax =plt.subplot(1,2,2)
ax=raw_df['Sex'].value_counts().plot.pie(explode=[0.1, 0.1],autopct='%1.2f%%',shadow=True);
ax.set_title(label = "Sex", fontsize = 16) #,color='Red',font='Lucida Calligraphy')
plt.savefig(pathIm + '/EDA43.png')
fig.clear(True)

fig, ax1 = plt.subplots()
heart=raw_df
ax1 = plt.subplot(1,2,1)
ax1 = sns.countplot(x='ChestPainType', data=heart)
ax1.bar_label(ax1.containers[0])
plt.title("ChestPainType", fontsize=14)
ax1 =plt.subplot(1,2,2)
ax1=heart['ChestPainType'].value_counts().plot.pie(explode=[0.1, 0.1,0.1,0.1],autopct='%1.2f%%',shadow=True);
ax1.set_title(label = "ChestPainType", fontsize = 20,color='Red',font='Lucida Calligraphy');
plt.savefig(pathIm + '/EDA44.png')
fig.clear(True)

fig, ax2 = plt.subplots()
ax2 = plt.subplot(1,2,1)
ax2 = sns.countplot(x='RestingECG', data=heart)
ax2.bar_label(ax2.containers[0])
plt.title("RestingECG", fontsize=14)

ax2 =plt.subplot(1,2,2)
ax2=heart['RestingECG'].value_counts().plot.pie(explode=[0.1, 0.1,0.1],autopct='%1.2f%%',shadow=True);
ax2.set_title(label = "RestingECG", fontsize = 20,color='Red',font='Lucida Calligraphy');
plt.savefig(pathIm + '/EDA45.png')
import time
time.sleep(1)
fig.clear(True)

fig, ax3 = plt.subplots()
ax3 = plt.subplot(1,2,1)
ax3 = sns.countplot(x='ExerciseAngina', data=heart)
ax3.bar_label(ax3.containers[0])
plt.title("ExerciseAngina", fontsize=14)

ax3 =plt.subplot(1,2,2)
ax3=heart['ExerciseAngina'].value_counts().plot.pie(explode=[0.1, 0.1],autopct='%1.2f%%',shadow=True);
ax3.set_title(label = "ExerciseAngina", fontsize = 20,color='Red',font='Lucida Calligraphy');
plt.savefig(pathIm + '/EDA46.png')
fig.clear(True)

fig, ax = plt.subplots()
ax = plt.subplot(1,2,1)
ax = sns.countplot(x='ST_Slope', data=heart)
ax.bar_label(ax.containers[0])
plt.title("ST_Slope", fontsize=14)

ax =plt.subplot(1,2,2)
ax=heart['ST_Slope'].value_counts().plot.pie(explode=[0.1, 0.1,0.1],autopct='%1.2f%%',shadow=True);
ax.set_title(label = "ST_Slope", fontsize = 20,color='Red',font='Lucida Calligraphy');
plt.savefig(pathIm + '/EDA47.png')
fig.clear(True)

sns.set(font_scale=1.1)
heart["Cholesterol_Category"]= pd.cut(heart["Cholesterol"] ,bins=[0, 200, 230 , 500] ,labels=["Normal","Borderline","High" ] )
print("Value Counts :\n\n",heart['Cholesterol_Category'].value_counts())

heart.head()
fig, ax = plt.subplots()
ax = plt.subplot(1,2,1)
ax = sns.countplot(x='Cholesterol_Category', data=heart)
ax.bar_label(ax.containers[0])
plt.title("Cholesterol_Categoryy", fontsize=14)

ax =plt.subplot(1,2,2)
ax=heart['Cholesterol_Category'].value_counts().plot.pie(explode=[0.1, 0.1,0.1],autopct='%1.2f%%',shadow=True);
ax.set_title(label = "Cholesterol_Category", fontsize = 20,color='Red',font='Lucida Calligraphy');
plt.savefig(pathIm + '/EDA48.png')
fig.clear(True)

heart["RestingBP_Category"]= pd.cut(heart["RestingBP"] ,bins=[0,120, 129 , 139,200] ,labels=["Normal_BP","Elevated_BP","Hypertension_Stage_1", "Hypertension_Stage_2"] )
print("Value Counts :\n\n",heart['RestingBP_Category'].value_counts())
heart.sample(5)
heart['RestingBP_Category'] = heart['RestingBP_Category'].astype(object)

plt.rcParams['legend.fontsize'] = 7
sns.set(font_scale=1.0)
fig, ax = plt.subplots()
ax = plt.subplot(1,2,1)
ax = sns.countplot(x='RestingBP_Category', data=heart)
ax.bar_label(ax.containers[0])
plt.axis('off');

ax =plt.subplot(1,2,2)
ax=heart['RestingBP_Category'].value_counts().plot.pie(explode=[0.1, 0.1,0.1,0.1],autopct='%1.2f%%',shadow=True);
plt.axis('off');
plt.savefig(pathIm + '/EDA49.png')
fig.clear(True)

df = heart  
male_df = df[df['Sex'] == 'M']
female_df = df[df['Sex'] == 'F']

## Grouping Datasets
male_cp_fbs = male_df.groupby(['ChestPainType', 'FastingBS']).size().reset_index().rename(columns={0: 'count'})
female_cp_fbs = female_df.groupby(['ChestPainType', 'FastingBS']).size().reset_index().rename(columns={0: 'count'})

male_st_ecg = male_df.groupby(['ST_Slope', 'RestingECG']).size().reset_index().rename(columns={0: 'count'})
female_st_ecg = female_df.groupby(['ST_Slope', 'RestingECG']).size().reset_index().rename(columns={0: 'count'})

male_ea_cp = male_df.groupby(['ExerciseAngina', 'ChestPainType']).size().reset_index().rename(columns={0: 'count'})
female_ea_cp = female_df.groupby(['ExerciseAngina', 'ChestPainType']).size().reset_index().rename(columns={0: 'count'})

## Creating Sunburst Figures
sb1 = px.sunburst(male_cp_fbs, values='count', path=['ChestPainType', 'FastingBS'])
sb2 = px.sunburst(female_cp_fbs, values='count', path=['ChestPainType', 'FastingBS'])

sb3 = px.sunburst(male_st_ecg, values='count', path=['ST_Slope', 'RestingECG'])
sb4 = px.sunburst(female_st_ecg, values='count', path=['ST_Slope', 'RestingECG'])

sb5 = px.sunburst(male_ea_cp, values='count', path=['ExerciseAngina', 'ChestPainType'])
sb6 = px.sunburst(female_ea_cp, values='count', path=['ExerciseAngina', 'ChestPainType'])

## Subplots
fig = make_subplots(rows=3, cols=2, specs=[
    [{"type": "sunburst"}, {"type": "sunburst"}],
    [{"type": "sunburst"}, {"type": "sunburst"}],
    [{"type": "sunburst"}, {"type": "sunburst"}]
], subplot_titles=("Male Chest Pain with Fasting Blood Sugar", "Female Chest Pain with Fasting Blood Sugar",
                   "Male ST Slope with Resting ECG", "Female ST Slope with Resting ECG",
                   "Male Exercise Angina with Chest Pain Type", "Female Exercise Angina with Chest Pain Type"))

## Plotting Figures
fig.add_trace(sb1.data[0], row=1, col=1)
fig.add_trace(sb2.data[0], row=1, col=2)
fig.add_trace(sb3.data[0], row=2, col=1)
fig.add_trace(sb4.data[0], row=2, col=2)
fig.add_trace(sb5.data[0], row=3, col=1)
fig.add_trace(sb6.data[0], row=3, col=2)

fig.update_traces(textinfo="label+percent parent")
fig.update_layout(title_text="Male vs Female Sunburst", title_x=0.5, 
                  height=1200, width=1200, template='plotly_dark', showlegend=False,
        font=dict(
            family="Rubik",
            size=14)
)

fig.write_image(pathIm + '/EDA50.png',scale=6)

heart_dft= heart 
RestingECG_vs_Sex = (
    heart_dft[["RestingECG", "Sex"]]
    .value_counts(normalize=True)
    .reset_index(name="Pct")
    .sort_values(by="RestingECG")
)
RestingECG_vs_Sex["Pct"] = RestingECG_vs_Sex["Pct"].round(2) * 100
RestingECG_vs_Sex.sort_values(by="Pct", ascending=False)

ChestPainType_vs_Sex = (
    heart_dft[["ChestPainType", "Sex"]]
    .value_counts(normalize=True)
    .reset_index(name="Pct")
    .sort_values(by="ChestPainType")
)
ChestPainType_vs_Sex["Pct"] = ChestPainType_vs_Sex["Pct"].round(2) * 100
plt.style.use("fivethirtyeight")
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
palette4 = {"ASY": "#1b85b8", "ATA": "#5a5255", "NAP": "#559e83", "TA": "#ae5a41"}
palette5 = {"LVH": "#2dc937", "Normal": "#e7b416", "ST": "#cc3232"}

sns.barplot(
    data=ChestPainType_vs_Sex,
    x="Sex",
    hue="ChestPainType",
    #errorbar=None,
    y="Pct",
    palette=palette4,
    linewidth=0.5,
    edgecolor="black",
    alpha=0.8,
    ax=ax[0],
)

for ax1 in [ax[0]]:
    for container in ax1.containers:
        values2 = container.datavalues
        labels = ["{:g}%".format(val) for val in values2]
        ax1.bar_label(container, labels=labels)

ax[0].set_ylabel("Percent")
ax[0].set_xlabel("")
ax[0].set_title(
    "Regardless of the proportion of Males and Females,\n Men have high ASY compared with Women, and the pattern is different.",
    fontsize=10,
)

sns.barplot(
    data=RestingECG_vs_Sex,
    x="Sex",
    hue="RestingECG",
    #errorbar=None,
    y="Pct",
    palette=palette5,
    linewidth=0.5,
    edgecolor="black",
    alpha=0.8,
    ax=ax[1],
)

for ax2 in [ax[1]]:
    for container in ax2.containers:
        values3 = container.datavalues
        labels = ["{:g}%".format(val) for val in values3]
        ax2.bar_label(container, labels=labels)

ax[1].set_ylabel("")
ax[1].set_xlabel("")
ax[1].set_title("Men and Women have somehow same pattern of RestingECG", fontsize=10)
plt.tight_layout()
fig.savefig(pathIm + '/EDA51.png')
fig.clear(True)

ExerciseAngina_vs_Sex = (
    heart_dft[["ExerciseAngina", "Sex"]]
    .value_counts(normalize=True)
    .reset_index(name="Pct")
    .sort_values(by="ExerciseAngina")
)
ExerciseAngina_vs_Sex["Pct"] = ExerciseAngina_vs_Sex["Pct"].round(2) * 100


ST_Slope_vs_Sex = (
    heart_dft[["ST_Slope", "Sex"]]
    .value_counts(normalize=True)
    .reset_index(name="Pct")
    .sort_values(by="ST_Slope")
)
ST_Slope_vs_Sex["Pct"] = ST_Slope_vs_Sex["Pct"].round(2) * 100

plt.style.use("fivethirtyeight")
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

palette6 = {
    "Y": "#000000",
    "N": "#009900",
}

palette7 = {"Down": "#b2d8d8", "Flat": "#66b2b2", "Up": "#004c4c"}

sns.barplot(
    data=ExerciseAngina_vs_Sex,
    x="Sex",
    hue="ExerciseAngina",
    #errorbar=None,
    y="Pct",
    palette=palette6,
    linewidth=0.5,
    edgecolor="black",
    alpha=0.8,
    ax=ax[0],
)

for ax3 in [ax[0]]:
    for container in ax3.containers:
        values2 = container.datavalues
        labels = ["{:g}%".format(val) for val in values2]
        ax3.bar_label(container, labels=labels)

ax[0].set_ylabel("Percent")
ax[0].set_xlabel("")
ax[0].set_title(
    "Almost a similar pattern between Men and Women. (ExerciseAngina)", fontsize=10
)

sns.barplot(
    data=ST_Slope_vs_Sex,
    x="Sex",
    hue="ST_Slope",
    #errorbar=None,
    y="Pct",
    palette=palette7,
    linewidth=0.5,
    edgecolor="black",
    alpha=0.8,
    ax=ax[1],
)

for ax4 in [ax[1]]:
    for container in ax4.containers:
        values3 = container.datavalues
        labels = ["{:g}%".format(val) for val in values3]
        ax4.bar_label(container, labels=labels)

ax[1].set_ylabel("")
ax[1].set_xlabel("")
ax[1].set_title(
    "A different pattern between Men and Women (ExerciseAngina)", fontsize=10
)
plt.tight_layout()
fig.savefig(pathIm + '/EDA52.png')
print(" ")
print(" ")

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
