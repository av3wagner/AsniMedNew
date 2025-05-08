import os, sys, inspect, time, datetime
import subprocess
import json
from time import time, strftime, localtime
from datetime import timedelta
import shutil
import os
import pandas as pd
from configparser import ConfigParser
config = ConfigParser()

now=datetime.datetime.now()
timestart = now.replace(microsecond=0)
print("Programm Start: ", timestart)
cwd = 'C:\AW75\AsniMed'
DirPr = cwd
DirData=os.path.join(cwd, "data")
DirAssets=os.path.join(cwd, "ASSETS")
DirImage=os.path.join(cwd, "Image")
DirImageNow=os.path.join(cwd, "Image")
DirTemplate=os.path.join(cwd, "Templates")
InEDA_File = os.path.join(cwd, "Templates\TemplateTOC1.docx")  
InEDA = os.path.join(cwd, "data\InEDA.docx")
ReportEDA = os.path.join(cwd, "OUTPUT-RESSOURCE\ReportEDA.docx")
ReportEDAr = os.path.join(cwd, "data\Kap3.docx")
rReportTest= os.path.join(cwd, "OUTPUT\TReportTest.docx")
ReportAll = os.path.join(cwd, "OUTPUT-RESSOURCE\ReportReportAll.docx")
excel_filename=os.path.join(cwd, "OUTPUT\EDA-Report5.xlsx") 
CONFIG_JSON = os.path.join(cwd, "Templates\ASNIR.json")
EDA_JSON    = os.path.join(cwd, "Templates\EDA2025.json")
tabdoc=os.path.join(cwd, "OUTPUT\EDA-Report.DOCX")

config['path'] ={
    'DirPr': DirPr,
    'DirData': DirData,
    'DirAssets': DirAssets,
    'DirImage': DirImage,
    'DirImageNow': DirImageNow,
    'DirTemplate': DirTemplate,
    'InEDA_File': InEDA_File,
    'InEDA': InEDA,
    'ReportEDA': ReportEDA,
    'ReportEDAr': ReportEDAr,
    'rReportTest': rReportTest, 
    'ReportAll': ReportAll,
    'excel_filename': excel_filename,
    'CONFIG_JSON': CONFIG_JSON,
    'EDA_JSON': EDA_JSON,
    'tabdoc': tabdoc,
}

with open('ProPath.ini', 'w') as output_file:
    config.write(output_file)

import configparser
config = configparser.ConfigParser()
config.read('ProPath.ini')
for section_name in config.sections():
    print ('Section:', section_name)
    print ('  Options:', config.options(section_name))
    for name, value in config.items(section_name):
        print ('  %s = %s' % (name, value))
    print("-----------------------------------")

DirPr   = config.get('path', 'dirpr')
pfad   = config.get('path', 'dirdata')
pathIm = config.get('path', 'DirImageNow')
eda_json = config.get('path', 'eda_json')
InEDA_File = config.get('path', 'ineda_file')
config_json = config.get('path', 'config_json')
DirTemplate = config.get('path', "dirtemplate")
DirImage = config.get('path', "dirimage") 
InEDA = config.get('path', 'ineda')
ReportEDA =  config.get('path', 'reporteda')
ReportEDAr = config.get('path', 'reportedar')
rReportTest = config.get('path', 'rreporttest')
ReportAll = config.get('path', 'reportall')
DirAssets = config.get('path', 'dirassets')
tabdoc = config.get('path', 'tabdoc')
excel_filename = config.get('path', 'excel_filename')

print("DirPr: ", DirPr) 
print("pfad: ", pfad) 
print("pathIm: ", pathIm)
print("EDA_json: ", eda_json) 
print("config_json: ", config_json)
print("ineda_file: ", InEDA_File)
print("ineda: ", InEDA)
print("reporteda: ",ReportEDA)
print("reportedar: ",ReportEDAr)
print("reportall: ", ReportAll)
print("tabdoc: ", tabdoc)
print("DirTemplate: ", DirTemplate)
print("DirImage: ", DirImage) 
print("DirAssets: ", DirAssets)
print("excel_filename: ", excel_filename)
print("rReportTest: ", rReportTest)
print("csv-file: ", pfad+'\heart.csv')
matrix = pfad + "/heard.csv"

try:
    raw_df = pd.read_csv(pfad+'\heart.csv')
except:
    raw_df =pd.read_csv(pfad+'\heart.csv')
    
df=raw_df
print()
print("-----------------------------------")
