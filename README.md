# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values

 
 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VARUNKUMAR K
RegisterNumber:  212219040173
*/

import pandas as pd
df=pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Semster 2/Intro to ML/Placement_Data.csv")
df.head()
df.tail()
df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()
df1.isnull().sum()
#to check any empty values are there
df1.duplicated().sum()
#to check if there are any repeted values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1["gender"] = le.fit_transform(df1["gender"])
df1["ssc_b"] = le.fit_transform(df1["ssc_b"])
df1["hsc_b"] = le.fit_transform(df1["hsc_b"])
df1["hsc_s"] = le.fit_transform(df1["hsc_s"])
df1["degree_t"] = le.fit_transform(df1["degree_t"])
df1["workex"] = le.fit_transform(df1["workex"])
df1["specialisation"] = le.fit_transform(df1["specialisation"])
df1["status"] = le.fit_transform(df1["status"])
df1
x=df1.iloc[:,:-1]
x
y = df1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.09,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
#liblinear is library for large linear classification
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
print(lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]))
*/
```

## Output:
Original data(first five columns):
![image](https://user-images.githubusercontent.com/88264052/175298790-9af4fe6f-9009-41b0-9e71-6638a1c16de1.png)
Data after dropping unwanted columns(first five):
![image](https://user-images.githubusercontent.com/88264052/175298818-7a8a581b-a848-4b1f-8b9e-a8ac4736d0cc.png)
Checking the presence of null values:
![image](https://user-images.githubusercontent.com/88264052/175298839-d5d218e2-6ba2-4037-9812-333da85a7aa9.png)

Checking the presence of duplicated values:

![image](https://user-images.githubusercontent.com/88264052/175298867-43ebe5b5-8dfb-4a45-ba37-39efca51b81e.png)
Data after Encoding:
![image](https://user-images.githubusercontent.com/88264052/175298891-9bdd81e9-4b89-40f7-8464-b1203e264546.png)
X Data:
![image](https://user-images.githubusercontent.com/88264052/175298923-2b76f866-25df-4c14-acb5-29627e57e3dd.png)

Y Data:
![image](https://user-images.githubusercontent.com/88264052/175298970-24dcf41f-a40b-427f-944e-8f6f0732e458.png)

Predicted Values:
![image](https://user-images.githubusercontent.com/88264052/175298998-a439c54a-16bd-4fcf-a3d2-218fa4fb85b2.png)

Accuracy Score:
![image](https://user-images.githubusercontent.com/88264052/175299034-1c6e345e-847a-4dbc-822d-7c91ebe86f50.png)

Confusion Matrix:
![image](https://user-images.githubusercontent.com/88264052/175299058-a906b135-1c92-440d-b664-66b0d9ca03c4.png)

Classification Report:
![image](https://user-images.githubusercontent.com/88264052/175299100-c950e1cc-fd61-4b4f-9ec8-26456c801311.png)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
