import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#importing the required libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('/home/oem/Documents/BBIT/3.2/AI/kmeans/details.csv')

wcss = {}
for k in range(1, 10):
    km = KMeans(n_clusters=k, init='k-means++', max_iter=1000)
    km = km.fit(df[['Height','Weight']])
    wcss[k] = km.inertia_

Model = KMeans(n_clusters=2 ,init="k-means++", max_iter=1000)
Model.fit_predict(df[['Height','Weight']])
df['Clusters'] = Model.labels_
df.head()



#starting of knn
feature_names=["Height","Weight"]
X=df[feature_names].values
y=df["Clusters"].values


#Spliting dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state = 0)

# Instantiate learning model (k = 3)
classifier = KNeighborsClassifier(n_neighbors=3)

# Fitting the model
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

pickle.dump(classifier, open('modelmeans.pkl','wb'))

