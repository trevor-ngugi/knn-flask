#importing the required libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np #working with arrays. 
import pandas as pd #data manipulation library that is necessary for every aspect of data analysis or machine learning.
from sklearn.preprocessing import LabelEncoder


##load dataset
bmi=pd.read_csv("details.csv")

#separating columns that do not have strings and int
Z= bmi
a= bmi['Gender']

##changing values from string to number
le = LabelEncoder()
Z['Gender'] = le.fit_transform(Z['Gender'])
a= le.transform(a)

#start of knn classifying
feature_names=["Height","Weight","Gender"]
X=Z[feature_names].values
y=Z["Index"].values

#Spliting dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state = 0)

# Instantiate learning model (k = 3)
classifier = KNeighborsClassifier(n_neighbors=3)

# Fitting the model
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

pickle.dump(classifier, open('modelfix.pkl','wb'))