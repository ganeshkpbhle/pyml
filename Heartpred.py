from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV as gs
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.svm import SVC
from sqlalchemy import true
from sklearn.tree import DecisionTreeClassifier as dt
#1.reading data from train.csv file
df=pd.read_csv('data.csv')
#getting info of train and test for null values,count of each category
# print(df.isnull().sum())
# print(df['pred_attribute'].value_counts())

#some columns holds ? removing those columns
indices=df[(df['ca']=='?') | (df['thal']=='?')].index
df.drop(indices,inplace=True)

#2.correlation
# sns.heatmap(df.corr(),annot=True)
# plt.show()
#correlation data cp,exang,olpeak closely related to pred_attribute

#3.Splitting Data for Training and testing
X_train=df.drop('pred_attribute',axis=1)
Y_train=df[['pred_attribute']]
x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,test_size=0.25,random_state=42)

#4.Hyper parameters for Support vector classification and Decision tree
paramSvc={
'kernel':['poly'],
'C':[100,10,1.0,0.1,0.001],
'gamma':[1,0.1,0.01,0.001,0.0001]
}
#,'rbf','sigmoid'
paramDt={
    'criterion':['gini','entropy'],
    'splitter':['best','random'],
    'min_samples_split':[3,5,7,9,11]
}
#5.Gridsearch for best hyper params
model_svc=SVC()
model_dt=dt()
clf=gs(model_svc,paramSvc,n_jobs=-1,verbose=3)
clf.fit(x_train,y_train)
print(clf.best_estimator_)
print(clf.best_params_)
print(clf.best_score_)