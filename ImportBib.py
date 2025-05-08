from IPython.core.display import HTML
from PIL import Image
from collections import Counter
from colorama import Fore, Style 
from configparser import ConfigParser
from copy import deepcopy
from datetime import timedelta
from docx import *
from docx import Document
from docx import Document, enum
from docx.enum.section import WD_ORIENT
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_BUILTIN_STYLE
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.table import WD_ROW_HEIGHT_RULE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT as WD_ALIGN_PARAGRAPH
from docx.oxml import *
from docx.oxml import OxmlElement, ns
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from docx.oxml.ns import qn
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.shared import Inches
from docx.shared import Pt, Mm, Cm, Inches, RGBColor
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph
from docx2pdf import convert
#from docxtpl import DocxTemplate, InlineImage
from lightgbm import LGBMClassifier
from matplotlib import *
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.offline import iplot
from plotly.subplots import make_subplots
from scikitplot.metrics import plot_roc_curve
from sklearn import datasets
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegression, Perceptron, RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn.metrics import RocCurveDisplay, roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report,f1_score,confusion_matrix,precision_score,recall_score,balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score #
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, log_loss
from sklearn.model_selection import cross_val_predict as cvp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve, ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.svm import SVC, LinearSVC, SVR, NuSVC
from sklearn.tree import DecisionTreeClassifier
from spire.doc import *
from spire.doc.common import *
from time import time, strftime, localtime
from win32com import client
from win32com.client import constants
from xgboost import XGBClassifier
import atexit
import colorama
import cufflinks as cf
import datetime as dt
import docx
import io
import json
import lightgbm as lgb
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import os, sys, inspect, time, datetime
import pandas as pd
import patchworklib as pw
import plotly as pl
import plotly as pplt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline
import plotly.offline as po
import random
import re
import scipy.stats as stats
import seaborn as sns
import shutil
import sklearn
import subprocess
import sys
import time
import warnings
import win32com.client
import win32com.client, time, pythoncom
import xgboost as xgb
sns.set()
warnings.filterwarnings("ignore")