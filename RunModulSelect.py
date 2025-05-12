import streamlit as st  
import streamlit.components.v1 as components
import time
from  PIL import Image
import numpy as np
import pandas as pd
import base64
import sys
import inspect, os
import pathlib
from os import listdir
from os.path import isfile, join
import glob
import os
import subprocess
import modules.ReadPath as m

import streamlit.components.v1 as components
from  PIL import Image
import numpy as np
import pandas as pd
import sys
import inspect, os
from os.path import isfile, join
import glob
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st  
import streamlit.components.v1 as components
from  PIL import Image
import numpy as np
import pandas as pd
import base64
import sys
import inspect, os
import pathlib
from os import listdir
from os.path import isfile, join
import glob


import streamlit as st  
import streamlit.components.v1 as components
from  PIL import Image
import numpy as np
import pandas as pd
import os, sys
from os import listdir
from os.path import isfile, join
import pathlib
import base64
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns

def execute_python_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            python_code = file.read()
            exec(python_code)
    except FileNotFoundError:
        st.markdown(f"Error: The file '{file_path}' does not exist.")

def select_file():
    parent_path = 'modules/programs'
    fileList = []
    fileList = listdir(parent_path)
    onlyfiles = [f for f in fileList if isfile(join(parent_path, f)) and  (f.endswith(".py"))]   
    option = st.selectbox('Выберите программу для исполнения', onlyfiles)
    file_location=os.path.join(parent_path, option) 
    if file_location.find('.py') > 0:
        st.write("Для исполнения выбрана программа: " + option)
        if st.button('Запустите выбранняю программу'):
            if option == "EDAReports.py":
                execute_python_file(file_location)
                st.write('Программа закончила работу!')
            else:   
                execute_python_file(file_location)
            
        if st.button('Покажите выбранняю программу'):    
            with open(file_location, 'r', encoding='utf-8') as f:
                 lines_to_display = f.read()
            st.code(lines_to_display, "python")    

st.set_page_config(
    page_title="Asfendijarov Kazakh National Medical University «АСНИ-МЕД»",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

col1, col2, col3 = st.columns( [1, 40, 1])
with col2:  
    st.markdown(f'<h1 style="color:yellow;font-size:36px;text-align:center">{"Asfendijarov Kazakh National Medical University"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:yellow;font-size:28px;text-align:center">{"Автоматизировання Обучающая Система Научных Исследований в медицине и здравоохранении «АСНИ-Обучение»"}</h1>', unsafe_allow_html=True)
    st.markdown("")
  
select_file()