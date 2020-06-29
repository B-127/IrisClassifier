# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:30:15 2020

@author: Tony

"""
"""
Detailed slice assignment.
[0]     #means line 0 of your matrix
[(0,0)] #means cell at 0,0 of your matrix
[0:1]   #means lines 0 to 1 excluded of your matrix
[:1]    #excluding the first value means all lines until line 1 excluded
[1:]    #excluding the last param mean all lines starting form line 1 included
[:]     #excluding both means all lines
[::2]   #the addition of a second ':' is the sampling. (1 item every 2)
[::]    #exluding it means a sampling of 1
[:,:]   #simply uses a tuple (a single , represents an empty tuple) instead of an index.
"""
#loading libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#loading dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset= read_csv(url,names=names)

#dataset shapes
print(dataset.shape)

#taking a peak into the data
print(dataset.head)

#describing the data. This includes the count,mean,the min and max values as
# well as some percentiles.
print(dataset.describe())

#class distribution. take a look at the number of instances (rows) that belong
# to each class. We can view this as an absolute count.
print(dataset.groupby("class").size())

#univariate plots- to better understand each attribute.
#using box and whisker plots
dataset.plot(kind="box", subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

#histograms
dataset.hist()
pyplot.show()

#multivariate plots- to better understand the relationships between attributes.
#scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

#Creating the test dataset. In this case 80% will be used for training while 20% will be used for testing
#https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
array=dataset.values
X=array[:,0:4]              #X is the input variables
Y=array[:,4]                #Y is the output variables
X_train, X_validation, Y_train, Y_validation= train_test_split(X,Y,test_size=0.2)
#You now have training data in the X_train and Y_train for preparing models and a 
#X_validation and Y_validation sets that we can use later.

#Test Harness
#Using stratified 10-fold cross validation to estimate model accuracy
#Cross-validation is a statistical method used to estimate the skill of machine learning models.
#https://machinelearningmastery.com/k-fold-cross-validation/

"""#Building Model
models=[]
models.append(('LR',LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

#evaluating each model
results=[]
names=[]

for name,model in models:
    kfold=StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results=cross_val_score(model,X_train,Y_train,cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s:%f(%f)'%(name,cv_results.mean(),cv_results.std()))
    
#comparing algorithms using box plots
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()
"""

#Making PREDICTIONS on test dataset
#https://machinelearningmastery.com/make-predictions-scikit-learn/
model=SVC(gamma='auto')
model.fit(X_train,Y_train)
predictions=model.predict(X_validation)

#Evaluating the model
#We can evaluate the predictions by comparing them to the expected results in the validation 
#set, then calculate classification accuracy, as well as a confusion matrix and a classification report.
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


    