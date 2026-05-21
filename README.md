# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and create the dataset for student placement prediction.
2. Split the dataset into training and testing data and apply feature scaling.
3. Train the SGD Classifier using logistic regression loss on the training dataset.
4. Predict the placement status for test data and evaluate the model performance.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: V G SAIRAIMA
RegisterNumber:  212225040359
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

data = {
    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9],
    'Previous_Score': [40, 50, 55, 60, 65, 70, 75, 80],
    'Internship': [0, 0, 1, 0, 1, 1, 1, 1],
    'Placement': [0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[['Hours_Studied', 'Previous_Score', 'Internship']]
y = df['Placement']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sgd_model = SGDClassifier(
    loss='log_loss',
    max_iter=1000,
    learning_rate='optimal',
    random_state=42
)

sgd_model.fit(X_train, y_train)

y_pred = sgd_model.predict(X_test)
y_prob = sgd_model.predict_proba(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

new_student = np.array([[6, 68, 1]])
new_student_scaled = scaler.transform(new_student)

placement_pred = sgd_model.predict(new_student_scaled)
placement_prob = sgd_model.predict_proba(new_student_scaled)

print(f"\nPredicted Placement Status: {'Placed' if placement_pred[0] == 1 else 'Not Placed'}")
print(f"Probability of Placement: {placement_prob[0][1]:.2f}")
```

## Output:
![image](http://github.com/VGSAIRAIMA/SGD-Classifier/blob/main/Screenshot%202026-05-21%20203410.png)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
