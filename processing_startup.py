import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 
from sklearn.model_selection import train_test_split
import math
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt  # Matlab-style plotting
import plotly.offline as py
#py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
sns.set(style='white', context='notebook', palette='deep')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
Random_state=42
np.random.seed(0)


#Models import
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
#import imputer:
from sklearn.impute import KNNImputer
#score
from sklearn.metrics import f1_score
from sklearn.ensemble import StackingClassifier

pd.set_option('display.max_columns', None)

dataset=pd.read_csv("C:/Users/IMEN/Mlops-project/startup_data.csv",\
                    converters={'status': lambda x: int(x == 'acquired')},parse_dates=['founded_at','first_funding_at','last_funding_at'])

dataset.rename(columns={'status':'is_acquired'}, inplace=True)



## Dropping unwanted features 

def process_dataset(dataset,n_neighbors=10):
    # Drop columns
    numerical_df_3=dataset.select_dtypes(numerics)
    columns_to_drop = ["Unnamed: 6", "Unnamed: 0"]
    dataset.drop(columns_to_drop, axis=1, inplace=True)
    comparison_column = np.where(dataset["state_code"] != dataset["state_code.1"], True, False)
    values_to_select = dataset.loc[comparison_column, 'state_code.1']
    dataset.drop(["state_code.1"], axis=1, inplace=True)
    numerical_column_names = dataset.select_dtypes([np.number]).columns
    knn= KNNImputer()
    knn_dataset= knn.fit_transform(dataset[numerical_column_names])
    
    dataset[numerical_column_names]=pd.DataFrame(knn_dataset)
    dataset['closed_at']=dataset['closed_at'].fillna('temporary')
    dataset['closed_at'] = dataset.closed_at.apply(lambda x: 1 if x =='temporary' else 0)
    #convert object_id to numeric:
    dataset['object_id'] = dataset['object_id'].str.replace("c:", '').astype(int)
    dataset['id'] = dataset['id'].str.replace("c:", '').astype(int)

    return dataset #, values_to_select
##Removing the columns "Unnamed: 0" and "Unnamed: 6", Unnamed: 6 
#has 493 missimg data as we dont have info about this cloumn. Unnamed:0 is unknown.
dataset  = process_dataset(dataset)
#Check for null values
#print(dataset.isnull().sum())

####################################### Imputing missing values with KNN Imputer: #################################################

#dataset=imputing_numeric_missing_values(dataset)

# Check for Null values
#print(dataset.isnull().sum())

############################# dealing with Time Series features###########################
#dealing with Time Series features
def time_series_data(dataset):
    dataset['months_between_first_and_last_funding'] = ((dataset.last_funding_at - dataset.first_funding_at)/np.timedelta64(1, 'M'))
    dataset['months_between_foundation_and_first_funding']=((dataset.first_funding_at - dataset.founded_at)/np.timedelta64(1, 'M'))
    #delete unnecessary data
    dataset.drop(["last_funding_at"],axis=1, inplace=True)
    dataset.drop(["first_funding_at"], axis=1, inplace=True)
    dataset.drop(["founded_at"], axis=1, inplace=True)
    return dataset


"""def draw_heatmap(dataset):

    f, ax = plt.subplots(figsize = (18, 18))
    
    corrMatt = dataset.corr(method='spearman')
    
    sns.heatmap(corrMatt, annot = True, linewidth = 0.5, fmt = '.1f', ax = ax)
    plt.show()"""

#draw_heatmap(numerical_df_4)



def drop_features(numerics, dataset):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_df_4=dataset.select_dtypes(numerics)
    # Create correlation matrix
    corr_matrix = numerical_df_4.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.loc["is_acquired"]
    upper=upper.fillna(0)
    upper=upper.to_dict()
    # Find features with correlation greater than 0.95
    to_drop = [key for key in upper if upper[key]< 0.2]
    dataset.drop(to_drop, axis=1, inplace=True)
    numerical_df_5=dataset.select_dtypes(numerics)
    dataset.drop(["labels"], axis=1, inplace=True)
    dataset.drop(["closed_at"], axis=1, inplace=True)
    dataset.drop(["months_between_first_and_last_funding"], axis=1, inplace=True) #corelated to founding_rounds
    numerical_df_5=dataset.select_dtypes(numerics)
    #draw_heatmap(numerical_df_5)
    dataset=pd.get_dummies(dataset)
    return dataset

