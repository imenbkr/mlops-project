import numpy as np # linear algebra
import pandas as pd 
from sklearn.model_selection import train_test_split
import math
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt  # Matlab-style plotting
import plotly.offline as py
py.init_notebook_mode(connected=True)
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

y=dataset["is_acquired"]
X= dataset.loc[:, dataset.columns != 'is_acquired']
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42)

######################################### Ensemble modelling #######################################

#voting Classifier:
 
votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best)
,('gbc',GBC_best)], voting='soft', n_jobs=-1)

votingC = votingC.fit(X_train, y_train)

#stacking
def stacking(classifiers, X_train, X_test, y_train, y_test):
    all_estimators = []
    for classifier in classifiers:
        all_estimators.append((str(classifier), classifier))
    stack = StackingClassifier(estimators=all_estimators, final_estimator=GBC_best)
    score= stack.fit(X_train, y_train).score(X_test, y_test)
   
    return score

    
test_is_acquired = pd.Series(votingC.predict(X_test), name="is_acquired")

results = pd.concat([test_is_acquired],axis=1)

score= f1_score(y_test, results, average='macro')

stacking_score = stacking(best_classifiers, X_train, X_test, y_train, y_test)

print(f'the voting score is: {score}')

print(f'the stacking score is: {stacking_score} ') 
