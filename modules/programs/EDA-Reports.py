#st.dataframe(data.head())

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
import matplotlib.pyplot as plt
import seaborn as sns

with open("assets/EDA01.jpg ", "rb") as img_file:    
        img01 = "data:image/png;base64," + base64.b64encode(img_file.read()).decode()
        
with open("assets/EDA02.jpg ", "rb") as img_file:    
        img02 = "data:image/png;base64," + base64.b64encode(img_file.read()).decode()   
        
columns=['age', 'operation_year', 'lymph_nodes', 'survival_status']
data=pd.read_csv("data/haberman.csv",names=columns)
    
def change_param(x):
    if(x==1):
        return 'yes'
    return 'no'

chVar=lambda x:change_param(x)
data["survival_status"]=pd.DataFrame(data.survival_status.apply(change_param))        

Part1 = '''## Что такое исследовательский анализ данных (EDA) и зачем он нам нужен?
В чем заключается концепция?

Исследовательский анализ данных - это набор методов, которые были в основном разработаны Джоном Тьюки, Джоном Уайлдером с 1970 года. Философия этого подхода заключается в изучении данных перед применением конкретной вероятностной модели. По словам Джона Тьюки и Джона Уайлдера, исследовательский анализ данных похож на детективную работу.

Исследовательский анализ данных (EDA ) был предложен Джоном Тьюки , чтобы побудить статистиков изучить данные и, возможно, сформулировать гипотезы, которые могли бы привести к сбору новых данных и экспериментам.

    “Величайшая ценность картины – это когда она заставляет нас заметить то, чего мы никогда не ожидали увидеть“ - Джон Тьюки 

EDA отличается от анализа исходных данных (IDA) , который более узко фокусируется на проверке допущений, необходимых для подгонки модели и проверки гипотез, а также на обработке недостающих значений и выполнении преобразований переменных по мере необходимости. EDA включает в себя IDA.
'''

Part2 = '''###### Вот некоторые обычно используемые графики для EDA:

*    Гистограммы: для проверки распределения определенной переменной
*    Точечные графики: для проверки зависимости между двумя переменными
*    Карты: для отображения распределения переменной на региональной или мировой карте.
*    График корреляции объектов (тепловая карта): для понимания зависимостей между несколькими переменными
*    Графики временных рядов: для определения тенденций и сезонности в данных, зависящих от времени 
'''

Part3 = '''######  “Это моя любимая часть аналитики: брать скучные плоские данные и воплощать их в жизнь с помощью визуализации “ - Джон Тьюки 

Потребность в Исследовательском анализе данных:

* 1. Визуально проанализируйте набор данных, чтобы получить представление о данных

* 2. Использование множества методов для того, чтобы сделать определенные выводы из полученных данных

* 3. Помогает нам в выборе функций, поскольку мы можем определить важные функции

* 4. Затем эти функции будут использованы для построения моделей машинного обучения

* 5. Цель состоит в том, чтобы определить, является ли прогнозирующая модель жизнеспособным аналитическим инструментом для конкретной бизнес-задачи, и если да, то какой тип моделирования наиболее подходит

* 6. Это помогает при передаче результатов нетехническому специалисту

* 7. Пропуск шага EDA может привести к неточному созданию модели

#### В качестве примера EDA нами используется набор данных Хабермана о выживаемости раковых пациентов 

Информация о наборе данных:

Описание: Набор данных содержит случаи из исследования, которое проводилось в период с 1958 по 1970 год в больнице Биллингса Чикагского университета по изучению выживаемости пациенток, перенесших операцию по поводу рака молочной железы.

Ссылка на набор данных: https://www.kaggle.com/gilsousa/habermans-survival-data-set
Artikel: https://aniketpatilvashi.medium.com/what-is-exploratory-data-analysis-and-why-we-need-it-3785254ac300

Атрибутивная информация:

Возраст пациента на момент операции (числовой)
Год операции пациента (1900 год, числовое значение)
Количество обнаруженных положительных подмышечных узлов (числовое значение)
Статус выживания (классовый атрибут):

1 = пациент прожил 5 лет или дольше .

2 = пациент умер в течение 5 лет .
'''

Part4='''##### Programm-Code
Импорт библиотек
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

загрузка файла
```python
df = pd.read_csv('haberman.csv')
df.head()
```

Вывод в файл первых 5 элементов

    30    64      1      1.1 
0   30    62      3      1 
1   30    65      0      1 
2   31    69      2      1 
3   31    65      4      1 
4   33    58      10     1 

Настройка соответствующих имен столбцов

```python
columns =['age','operation_year','positive_nodes','survival_status']
df.columns = columns
df.head()
```

Файл вывода первых 5 элементов с обновленными именами столбцов

    возраст операции_год положительных_нод выживания_статус 
0   30            62             3             1 
1   30            65             0             1 
2   31            69             2             1 
3   31            65             4             1 
4   33            58             10            1 

Изменение значения целевого столбца

##### измените значения целевого столбца на 1 и 0 вместо 1 и 2, где 1: выживший и 0: умерший 

```python
# modify the target column values as 1 and 0 instead of 1 and 2  where 1 : alive  and 0 :dead
df['survival_status'] = df['survival_status'].map({1:1, 2:0})
print(df.head())
```

##### **Проверьте размерность набора данных**

```python
df.форма (305, 4) 
```

##### **Определите количество выживших и умерших пациентов**
```python
df['survival_status'].value_counts()

1    224
0     81
Name: survival_status, dtype: int64
```

Замечание:

Число пациентов, выживших после операции через 5 лет, составляет 224, а умерших до 5 лет - 81.

df.info () 
<класс 'pandas.core.frame.Фрейм данных'> 
Индекс диапазона: 305 записей, от 0 до 304 
Столбцы данных (всего 4 столбца): 
 ### Столбец с ненулевым числом Dtype 
---  ------           --------------  ----- 
 0 возраст 305 ненулевой int64 
 1 operation_year 305 ненулевой int64 
 2 положительных узла 305 ненулевых int64 
 3 survival_status 305 ненулевой int64 
типы данных: int64(4) 
объем используемой памяти: 9,7 КБ 

Замечание:

все значения не равны нулю
Концепция — Процентное увеличение или процентное изменение.

a — b - это разница между a и b или изменение перехода от b к a.

Допустим, я думаю о том, что "a" больше, чем "b", так что эта величина имеет значение b, а затем увеличивается до значения a, следовательно, увеличение равно a - b.

Например, рассмотрим b = 12 и a = 18, тогда a — b = 6. Деление на b дает меру этого увеличения относительно значения b. В этом случае (a — b) /b = 6/12 = 1/2, или, выраженный в процентах, 50%.

Это величина, которую мы называем процентным увеличением или процентным изменением.
Процентное увеличение или процентное изменение

### давайте рассчитаем среднее значение и процентное изменение 
```python
# let us calculate mean and percentage change
print(df['survival_status'].mean())      # this can also be calcualted as follows : print((224)/305)
print((224-81)/(224))                    # Percentage change0.7344262295081967
0.6383928571428571
```
Базовая статистическа о данных

```python
df.describe()
```
#### Результат:
'''

Part4A='''##### Замечание:
После 5 лет операции шансов выжить на 73% больше, чем умереть. Это может указывать на несбалансированность набора данных, поскольку выживших более 75% (IQR). Так ли это? Мы узнаем позже.

У 75% пациентов имеется не более 4 положительных узлов, у 25% отсутствуют какие-либо положительные узлы, в то время как у 50% из них имеется по крайней мере 1 положительный узел

положительные узлы распределены неравномерно. Могут иметь выбросы; Или большая часть данных ближе к минимальному значению, поскольку среднее значение ближе к минимальному значению.
Определение общего количества за каждый операционный год

df['operation_year'].value_counts() 58 36 
64    30 
63    30 
66    28 
65    28 
60    28 
59    27 
61    26 
67    25 
62    23 
68    13 
69    11 
Имя: operation_year, dtype: int64 

Замечание:

58-й год эксплуатации показывает наибольшее количество операций, а 69-й - наименьшее
Расчет временных рамок для большинства операций

#в течение многих лет (61,62,63) <-- 64 -- > (65,66,67) 

26+23+30+30+28+28+25  # операций в год за конкретный год, указанный в приведенной выше строке 190 #За годы (59,60,61) <-- 62 -- > (63,64,65) Итак, 1962 год со стандартным отклонением 3 

27+28+26+23+30+30+28  # операции в год за конкретный год, указанный в приведенной выше строке 192 

Замечание:

Большинство операций выполнено в 1962 году с std-dev 3 года
Диапазон расчета для большинства операций в соответствии с возрастной группой

### (49,50,51) <-- 52 --> (53,54,55) 
### Это операции по возрасту для конкретного возраста, указанного в приведенной выше строке 

10+12+6+14+11+13+10  76  # (51,52,53) <-- 54 --> (55,56,57) # операции по возрасту для конкретного возраста, указанного в приведенной выше строке 


6+14+11+13+10+7+11 72 

Замечание:

Существуют различные возрастные группы, которым проводится операция

большинство операций было проведено в возрастной группе 52 года со стандартным отклонением в 3 года

### МНОГОМЕРНЫЙ АНАЛИЗ (MULTIVARIATE ANALYSIS)

По сути, многомерный анализ - это инструмент для поиска закономерностей и взаимосвязей между несколькими переменными одновременно. Он позволяет нам предсказать, какое влияние изменение одной переменной окажет на другие переменные. Это дает многомерному анализу решающее преимущество перед другими формами анализа.

```python
# First lets do multivariate analysis which will help us in univariate analysis
# Pair plotssns.set_style("whitegrid");
sns.pairplot(df, hue="survival_status", height=3);
plt.show()
```
'''

Part5='''##### Замечание:

Сбивающий с толку, давайте перейдем к одномерному анализу.

### Одномерный анализ (Univariate Analysis:):

Основная цель одномерного анализа - описать, обобщить и найти закономерности в одном объекте.
Функция плотности вероятности (PDF)

Функция плотности вероятности (PDF) - это вероятность того, что переменная примет значение x. (сглаженная версия гистограммы) Здесь высота столбика обозначает процент точек данных, относящихся к соответствующей группе

```python
# ВОЗРАСТ 
plt.close();
sns.FacetGrid(df, hue="survival_status", height=5) \
   .map(sns.distplot, "age") \
   .add_legend();
plt.show(); 
```
'''

Part6='''##### Замечание:
Замечание:

Это нормальное распределение с большим разбросом. В основном разнообразное? Да, но нормальное распределение

Наблюдается значительное совпадение, которое говорит нам о том, что шансы на выживание не зависят от возраста человека.

Хотя есть совпадения, мы можем смутно сказать, что люди, чей возраст находится в диапазоне 30-40 лет, с большей вероятностью выживут, а 40-60 лет - с меньшей вероятностью. В то время как люди, чей возраст колеблется в пределах 60-75 лет, имеют равные шансы выжить и не выжить.

Шансы на выживание не могут определяться исключительно возрастным фактором.

```python
# Год операции
plt.close();
sns.FacetGrid(df, hue="survival_status", height=5) \
   .map(sns.distplot, "operation_year") \
   .add_legend();
plt.show()
```
'''

Part7='''##### Замечание:

Наблюдается значительное совпадение. На этом графике показано только, сколько операций были успешными, а сколько - нет. Этот параметр не может определять шансы пациента на выживание.

Однако можно заметить, что в 1959 и 1965 годах было больше неудачных операций.

Замечание:

Уровень смертности имеет незначительную тенденцию к снижению. Является ли это следствием несбалансированности данных? Нет, он нормализован. Что может быть причиной этого? Это не наша цель прямо сейчас.

Наименьший показатель выживаемости в 1965 году. У людей, оперированных в 1965 году, меньше шансов на выживание по сравнению с 1961 и 67 годами. У пациентов 1963 года шансы на выживание немного выше.

Год работы - это хорошая функция для задачи классификации: попробуйте найти корреляцию между выжившим col и sur_ratio_normalized после сопоставления sur_ratio_normalized с дискретными значениями

```python
# positive_nodes: Должно быть интересно. Но надо иметь в виду, что это ВСЕГО лишь признак заболеваемости раком 

plt.close();
sns.FacetGrid(df, hue="survival_status", height=8) \
   .map(sns.distplot, "positive_nodes") \
   .add_legend();
plt.show();
```
'''

Part8='''##### Замечание:

Люди без большего количества положительных индексов после операции, как правило, выживают дольше. Интуитивно это верно, поскольку у них, возможно, даже не было рака!

Люди с большим количеством положительных узлов при операции, как правило, относительно чаще умирают и меньше выживают. Может ли это противоречие быть следствием несбалансированного набора данных? Нет, этого не должно быть.

Это означает, что positive_nodes определенно указывает на наличие рака, Но это подразумевает, что операции на молочной железе не очень полезны. Это приносит пользу только медицинским организациям, поскольку они могут проводить операции пациентам без рака, и пациент все равно выживает.

Для любого заданного числа положительных узлов (за исключением близких к нулю) Выживаемость всегда меньше по сравнению с не-выживаемостью!!!

Примечание: Всякий раз, когда возникает противоречие, смотрите на наличие смещений, выбросов и т.д. В данных

```python
plt.close();
sns.FacetGrid(df, hue="survival_status", height=5) \
   .map(sns.distplot, "survival_status") \
   .add_legend();
plt.show();
```
'''

Part9='''##### Замечание:

Это не может быть сильно несбалансированным набором данных, поскольку не-выживание составляет более половины планки выживания. (63% от этого, если быть точным = относительное соотношение) Это отвечает на наш запрос о неопределенности несбалансированного набора данных, которым он не является.

Замечание:

Полная ошибка неправильной классификации. У большинства людей максимум ~ 26 положительных узлов.

Одна вещь, которую мы можем отметить в противном случае, это то, что (за исключением случаев, близких к 0-4,6) Выживаемость всегда меньше по сравнению с не-выживаемостью!!!

у 83,55% выживших пациентов были узлы в диапазоне 0-4,6
#### Box Plots

Прямоугольник простирается от нижнего квартиля к верхнему значениям данных с линией по медиане. Усики простираются от прямоугольника, чтобы показать диапазон данных. Точки выброса - это те, которые находятся за концом усов.

Замечание:

Интересно, что выбросы положительных узлов ограничены, но почти у всех, кто выжил более чем через 5 лет после операции, максимальное количество положительных узлов составляло от 7 до 8

Медиана положительных значений для выживших пациентов равна нулю. Это центральная величина, которую мы выполняем EDA, удаляя выбросы для выживших пациентов.

#### Violin Plots

Violin Графики - это комбинация Box Plots и функции плотности вероятности (CDF).

Замечание:

Пациенты с более чем 1 узлом имеют меньшие шансы на выживание. Чем больше узлов, тем меньше шансов на выживание.

У большого процента выживших пациентов было 0 узлов. Тем не менее, небольшой процент пациентов, у которых не было положительных подмышечных узлов, умерли в течение 5 лет после операции, таким образом, отсутствие положительных подмышечных узлов не всегда гарантирует выживание.

Было сравнительно больше людей, которые были прооперированы в 1965 году и не прожили более 5 лет.

В возрастной группе от 45 до 65 лет было сравнительно больше людей, которые не выжили. Сам по себе возраст пациента не является важным параметром при определении выживаемости пациента.

Графики box и violin для параметров age и year дают аналогичные результаты со значительным перекрытием точек данных. Перекрытие на графике прямоугольника и графике скрипки узлов меньше по сравнению с другими характеристиками, но перекрытие все еще существует, и поэтому трудно установить пороговое значение для классификации обоих классов пациентов.
3D-график:
2D График плотности, Контурный график:

Заключительные замечания:

Нормальное распределение не может быть эффективно использовано для классификации, поскольку большее или меньшее количество пациенток из групп того же года пережили или не пережили операцию на молочной железе. Мы можем наблюдать пик смертности в 1964 году. Мы также можем видеть, что выживаемость со временем снижается.

Давайте сравним приведенный выше пункт с "уровнем выживаемости" или "смертностью", поскольку мы не можем зависеть только от данных о выживаемости или смертности:

Коэффициент выживаемости имеет незначительную тенденцию к снижению. Является ли это следствием несбалансированности данных? Нет, он нормализован. Что может быть причиной этого? Не наша цель.

Positive_nodes определенно указывает на наличие рака, Но это подразумевает, что операции на молочной железе не очень полезны.

Это не может быть сильно несбалансированным набором данных, поскольку отсутствие выживаемости составляет более половины показателя выживаемости

Поскольку это не несбалансированный набор данных, из этого набора данных можно сделать различные выводы как с помощью EDA, так и с помощью IDA
Заключение:

Вы можете диагностировать pак у пациентов, используя набор данных Хабермана, применяя различные методы анализа данных и используя различные библиотеки Python.
'''

with st.expander("Пример проведения исследовательского анализа данных (Exploratory Data Analysis)"):
    st.markdown("")
    st.markdown(Part1)
    st.markdown("")
    
    st.write(f"""
        <div class="container">
            <div class="box">
                <div class="spin-container">
                    <div class="shape">
                        <div class="bd">
                            <img src="{img01}" alt="AW" width="600" height="400" style="display: block; margin: auto">
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """, 
    unsafe_allow_html=True)
    st.markdown("")
    st.write(f"""
        <div class="container">
            <div class="box">
                <div class="spin-container">
                    <div class="shape">
                        <div class="bd">
                            <img src="{img02}" alt="AW" width="600" height="400" style="display: block; margin: auto">
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """, 
    unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown(Part3)
    st.markdown("")
    st.markdown(Part4)
    st.markdown("")
    
    st.markdown("")
    st.header("Базовая статистическа о данных")
    st.write(data.describe())   

    st.markdown("")
    st.markdown(Part4A)
    st.markdown("")    
 
    sns.set_style("whitegrid")
    st.header("Histogram for Survival status")    
    fig=plt.figure(figsize=(9,5)) 
    fig1=sns.pairplot(data,hue="survival_status", height=3)
    plt.rc('font', size=7)
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot(fig1)   
     
    st.markdown("")
    st.markdown(Part5)
    st.markdown("")
    
    ###### AGE
    st.header("Histogram for Age")    
    fig=plt.figure(figsize=(9,5))
    fig2=(sns.FacetGrid(data,hue="survival_status",height=5) \
             .map(sns.distplot,"age"))
    plt.title("Histogram for Age")
    plt.xlabel('AGE')
    plt.ylabel('VALUE')
    plt.legend()
    plt.rc('font', size=5)
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot(fig2)  
       
    st.markdown("")
        
    st.markdown(Part6)
    st.markdown("")
    
    #### histogram for operation_year
    st.header("Histogram for Operation Year")    
    fig=plt.figure(figsize=(9,5))
    fig3=(sns.FacetGrid(data,hue="survival_status",height=5)    \
        .map(sns.distplot,"operation_year"))
    plt.title("Histogram for Operation Year")
    plt.xlabel('OPERATION YEAR')
    plt.ylabel('VALUE')
    plt.legend()
    plt.rc('font', size=5)
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot(fig3)
    st.markdown("")
    
    st.markdown("")
    st.markdown(Part7)
    st.markdown("")
    
    #### histogram for lymph_nodes
    st.header("Histogram for Lymph Nodes")    
    fig=plt.figure(figsize=(9,3))
    #,height=5
    fig4=(sns.FacetGrid(data,hue="survival_status")    \
             .map(sns.distplot,"lymph_nodes"))
    plt.title("Histogram for Lymph Nodes")
    plt.legend()
    plt.xlabel('LYMPH NODES')
    plt.ylabel('VALUE')
    plt.rc('font', size=5)
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot(fig4)
    st.markdown("")
    
    st.markdown("")
    st.markdown(Part8)
    st.markdown("")
    
    
    #1 plotting the PDF and CDF for age
    st.markdown("")
    st.header("Plotting the PDF and CDF for age")    
    fig=plt.figure(figsize=(9,5))  
    cnt,bin_edges=np.histogram(data['age'],bins=10,density=True)
    pdf=cnt/sum(cnt)
    # compute CDF
    cdf=np.cumsum(pdf)
    plt.plot(bin_edges[1:],pdf,label="PDF")
    plt.plot(bin_edges[1:],cdf,label="CDF")
    plt.xlabel("Age")
    plt.ylabel("Probability")
    plt.title("PDF and CDF of Age")
    plt.legend()
    plt.rc('font', size=14)
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot(plt)
    
    #2 plotting the PDF and CDF for lymph_nodes
    st.markdown("")
    st.header("Plotting the PDF and CDF for lymph_nodesr")    
    fig=plt.figure(figsize=(9,5))  
    cnt,bin_edges=np.histogram(data['lymph_nodes'],bins=10,density=True)
    # calculating pdf
    pdf=cnt/sum(cnt)
    # compute CDF
    cdf=np.cumsum(pdf)
    plt.plot(bin_edges[1:],pdf,label="PDF")
    plt.plot(bin_edges[1:],cdf,label="CDF")
    plt.xlabel("Lymph nodes")
    plt.ylabel("Probability")
    plt.title("PDF and CDF of Lymph Nodes")
    #plt.legend()
    plt.legend('')
    plt.rc('font', size=14)
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot(plt)

    #3 plotting the PDF and CDF for operation_year
    st.markdown("")
    st.header("PDF and CDF of Operation Year")    
    fig=plt.figure(figsize=(9,5))  
    cnt,bin_edges=np.histogram(data['operation_year'],bins=10,density=True)
    pdf=cnt/sum(cnt)
    # compute CDF
    cdf=np.cumsum(pdf)
    plt.plot(bin_edges[1:],pdf,label="PDF")
    plt.plot(bin_edges[1:],cdf,label="CDF")
    plt.xlabel("Operation Year")
    plt.ylabel("Probability")
    plt.title("PDF and CDF of Operation Year")
    #plt.legend()
    plt.legend('')
    plt.rc('font', size=14)
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot(plt)
    st.markdown('''Наблюдение:
    Почти у 80% пациентов меньше или почти 6 лимфатических узлов
    ''')

    st.markdown("")
    st.markdown(Part9)
    st.markdown("")
    
    #4 box plot for age vs survival status
    st.markdown("")
    st.header("Survival Status vs Age")    
    fig=plt.figure(figsize=(9,5))  
    sns.boxplot(x='survival_status',y="age",data=data,hue="survival_status")
    plt.title("Survival Status vs Age")    
    plt.ylabel("Age")
    plt.xlabel("Survival Status")
    #plt.legend(loc="center")
    plt.legend('')
    plt.rc('font', size=14)
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot(plt)
    
    #5 box plot for op year vs survival status
    st.markdown("")
    st.header("Survival Status vs Operation year")    
    fig=plt.figure(figsize=(9,5))  
    sns.boxplot(x='survival_status',y="operation_year",data=data,hue="survival_status")
    plt.title("Survival Status vs Operation year")    
    plt.ylabel("Operation Year")
    plt.xlabel("Survival Status")
    #plt.legend(loc="center")
    plt.legend('')
    plt.rc('font', size=14)
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot(plt)
        
    #6 box plot for lymph nodes vs survival status
    st.markdown("")
    st.header("Survival Status vs Lymph Nodes")    
    fig=plt.figure(figsize=(9,5))   
    sns.boxplot(x='survival_status',y="lymph_nodes",data=data,hue="survival_status")
    plt.title("Survival Status vs Lymph Nodes")    
    plt.ylabel("Lymph Nodes")
    plt.xlabel("Survival Status")
    #plt.legend(loc="center")
    plt.legend('')
    plt.rc('font', size=14)
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot(plt)
                
    #7 violin plot for age
    st.markdown("")
    st.header("Survival Status vs Age")    
    fig=plt.figure(figsize=(9,5))   
    sns.violinplot( x='survival_status', y="age", data=data,hue="survival_status") #, height=2, aspect=2)
    plt.title("Survival Status vs Age")
    plt.xlabel("Survival Status")
    plt.ylabel("Age")
    #plt.legend(loc="center")
    plt.legend('')
    plt.rc('font', size=14)
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot(plt)
        
    st.markdown("")
    st.header("Survival Status vs Operation Year")    
    fig=plt.figure(figsize=(9,5))    
    #8 violin plot for op year
    sns.violinplot( x='survival_status', y="operation_year",data=data,hue="survival_status", height=3, aspect=2)
    #plt.title("Survival Status vs Operation Year")
    plt.xlabel("Survival Status")
    plt.ylabel("Age")
    plt.legend('')
    plt.rc('font', size=14)
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot(plt)
    
    
    st.markdown("")
    st.header("Survival Status vs Lymph Nodes")
    fig=plt.figure(figsize=(9,5))
    #f=sns.violinplot( x='survival_status', y="lymph_nodes", data=data, hue="survival_status", scale='width', height=5, aspect=2)
    f = sns.violinplot(data=data, x="survival_status", y="lymph_nodes", native_scale=True, inner='quartile', dodge=False, height=5, aspect=2)
    plt.title("Survival Status vs Lymph Nodes")
    plt.xlabel("Survival Status")
    plt.ylabel("lymph nodes")
    plt.legend('')
    plt.rc('font', size=14)
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot(plt)   

    
    st.markdown("")
    st.header("Vbar Chart")
    fig=plt.figure(figsize=(9,7))
    g=sns.catplot(x="age", y="lymph_nodes", data=data, kind="bar", height=5, aspect=2)
    col1, col2, col3,= st.columns([1, 7, 1])
    with col2:
        st.pyplot(g)   

        

    