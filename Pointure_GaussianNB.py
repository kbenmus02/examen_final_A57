import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

df = pd.read_csv('pointure.data')


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
input_classes = ['masculin','féminin']
label_encoder.fit(input_classes)

# transformer un ensemble de classes
encoded_labels = label_encoder.transform(df['Genre'])
print(encoded_labels)
df['Genre'] = encoded_labels

X = df.iloc[:, lambda df: [1, 2, 3]]
y = df.iloc[:, 0]

from sklearn.model_selection import train_test_split

#decomposer les donnees predicteurs en training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

import mlflow
import mlflow.sklearn
from sklearn import metrics
from urllib.parse import urlparse
with mlflow.start_run(nested=True):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    
    y_naive_bayes1 = gnb.predict(X_train)
    print("Number of mislabeled points out of a total 0%d points : 0%d" % (X_train.shape[0],(y_train != y_naive_bayes1).sum()))
    
    

    accuracy = metrics.accuracy_score(y_train, y_naive_bayes1)
    print("Accuracy du modele Naive Bayes predit: " + str(accuracy))


    recall_score = metrics.recall_score(y_train, y_naive_bayes1)
    print("recall score du modele Naive Bayes predit: " + str(recall_score))

    f1_score = metrics.f1_score(y_train, y_naive_bayes1)
    print("F1 score du modele Naive Bayes predit: " + str(f1_score))
    print("")
    #######################################################################################################################
    # evaluation sur le test
    y_naive_bayes2 = gnb.predict(X_test)
    print("Number of mislabeled points out of a total 0%d points : 0%d" % (X_test.shape[0],(y_test != y_naive_bayes2).sum()))

    recall_score_test_set = metrics.recall_score(y_test, y_naive_bayes2)
    print("recall score du modele Naive Bayes predit: " + str(recall_score_test_set))

    f1_score_test_set = metrics.f1_score(y_test, y_naive_bayes2)
    print("F1 score du modele Naive Bayes predit: " + str(f1_score_test_set))
    
    
    # enregistrer les metrics
    mlflow.log_metric("recall_score sur test_set", recall_score_test_set)
    mlflow.log_metric("f1_score sur test_set", f1_score_test_set)
    
    
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    
    mlflow.sklearn.log_model(gnb, "modele")
#     mlflow.end_run()