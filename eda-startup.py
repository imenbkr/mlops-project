import numpy as np # linear algebra
import pandas as pd 
from sklearn.model_selection import train_test_split
import math
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt  # Matlab-style plotting
import plotly.offline as py
import plotly.graph_objs as go
#py.plot(fig, filename='C:/Users/IMEN/Mlops-project/plot.html')
import plotly.tools as tls
sns.set(style='white', context='notebook', palette='deep')
import warnings


def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
Random_state=42
np.random.seed(0)

pd.set_option('display.max_columns', None)

dataset=pd.read_csv("C:/Users/IMEN/Mlops-project/startup data.csv",\
                    converters={'status': lambda x: int(x == 'acquired')},parse_dates=['founded_at','first_funding_at','last_funding_at'])
#print(dataset.head())
#print(dataset.info())
#print(dataset.columns)
#print(dataset.duplicated().value_counts())

dataset.rename(columns={'status':'is_acquired'}, inplace=True)
#print(dataset.describe())

######################################## Correlation between numeric parameters  #########################################

def draw_heatmap(dataset):
    
    
    f, ax = plt.subplots(figsize = (18, 18))
    
    corrMatt = dataset.corr(method='spearman')
    
    sns.heatmap(corrMatt, annot = True, linewidth = 0.5, fmt = '.1f', ax = ax)
    plt.show()
    
    
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
numerical_df_1=dataset.select_dtypes(numerics)
numerical_column_names = dataset.select_dtypes(numerics).columns

#draw_heatmap(numerical_df_1)

############################################### Detect outliers from numeric variables  ###############################################
def getOutliersMatrix(numerical_df, threshold=1.5):
    Q1 = numerical_df.quantile(0.25)
    Q3 = numerical_df.quantile(0.75)
    IQR = Q3 - Q1
    
    outdata = (numerical_df < (Q1 - 1.5 * IQR)) | (numerical_df > (Q3 + 1.5 * IQR))
    
    for name in numerical_df.columns:
        outdata.loc[(outdata[name] == True), name] = 1
        outdata.loc[(outdata[name] == False), name] = 0
    
    return outdata
outliersMatt = getOutliersMatrix(numerical_df_1)
outliersMatt = getOutliersMatrix(numerical_df_1)

dataset[outliersMatt==1]= np.nan

numerical_df_2=dataset.select_dtypes(numerics)

#draw_heatmap(numerical_df_2)

sns.histplot(data=dataset, x='funding_total_usd', bins=30, kde=True)
plt.xlabel('Funding Total (USD)')
plt.ylabel('Frequency')
plt.title('Distribution of Funding Total')
#plt.show()

plt.figure(figsize=(12, 6)) 

sns.countplot(data=dataset, x='state_code')

plt.xlabel('State Code')
plt.ylabel('Count')
plt.title('Count of Startups by State Code')
plt.xticks(rotation=45)
#plt.show()

sns.boxplot(data=dataset, x='is_top500', y='funding_total_usd')
plt.xlabel('Is in Top 500')
plt.ylabel('Funding Total (USD)')
plt.title('Funding Total by Top 500 Status')
#plt.show()

sns.scatterplot(data=dataset, x='age_first_funding_year', y='funding_total_usd')
plt.xlabel('Age at First Funding Year')
plt.ylabel('Funding Total (USD)')
plt.title('Relationship between Age at First Funding and Funding Total')
#plt.show()

