from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV as gs
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.svm import SVC

#1.reading data from train.csv file
df=pd.read_csv('train.csv')
#getting info of train and test for null values
#print(df.isnull().sum())
#print(test.isnull().sum())

#2.correlation
#sns.heatmap(df.corr())
#plt.show()
#correlation data shows that ram memory more correlated to mobile pricing

#3.Splitting Data for Training and testing
X_train=df.drop('price_range',axis=1)
Y_train=df[['price_range']]
x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,test_size=0.4,random_state=500)

#scaling data down
#std=StandardScaler()
#X=std.fit_transform(x_train)
#test_data=std.transform(x_test)

#4.Hyper parameter Tuning for Support Vector Machine
param={
'kernel':['poly','rbf','sigmoid'],
'C':[100,10,1.0,0.1,0.001],
'gamma':[1,0.1,0.01,0.001,0.0001]
}
#The Best Hyper parameter is
#SVC(C=100, gamma=1, kernel='poly')
#{'C': 100, 'gamma': 1, 'kernel': 'poly'}

model=SVC(C=100, gamma=1, kernel='poly',random_state=10)

#5.Grid Search using the hyper Parameters
#clf=gs(model,param,refit=True,verbose=3)
#clf.fit(x_train,y_train)
#print(clf.best_estimator_)
#print(clf.best_params_)
#print(clf.best_score_)

#6.Training Model
model.fit(x_train,y_train)

#7.predicting based on test data
res=model.predict(x_test)
res_train=model.predict(x_train)
print("Accuracy on testing:"+str(accuracy_score(res,y_test)*100))
print("Confusion matrix:")
print(confusion_matrix(res,y_test))
print("Classification Details:")
print(classification_report(res,y_test))