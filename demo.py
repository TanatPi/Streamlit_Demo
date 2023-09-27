import streamlit as st
import pickle
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets

iris = datasets.load_iris() # sklearn.utils._bunch.Bunch type
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target']) # convert to dataframe

keys = np.array([0,1,2])
dictionary = {keys[i]: iris.target_names[i] for i in range(len(keys))}

df["target"].replace(dictionary, inplace = True)

X = df.drop(columns = 'target')
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 1)

classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

score = accuracy_score(y_test,y_pred)

pickle_out = open("model_iris.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()