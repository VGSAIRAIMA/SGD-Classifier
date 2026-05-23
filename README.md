# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the Iris flower dataset containing feature values and target species.
2. Split the dataset into training and testing data for model development.
3. Train the SGD Classifier using the training dataset to classify the Iris flower species.
4. Predict the species for the test data and evaluate the model using accuracy score and confusion matrix.
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: V G SAIRAIMA
RegisterNumber:  212225040359
*/
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: R . Nithish Aaditiyaa
RegisterNumber:  25011876[ 212225040287 ]
*/

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

df['target'] = iris.target
print(df.head())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train)
y_pred = sgd_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
cm = confusion_matrix(y_test, y_pred)
print("confusion_matrix:")
print(cm)
```

## Output:
![image](https://github.com/VGSAIRAIMA/SGD-Classifier/blob/main/Screenshot%202026-05-21%20203410.png)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
