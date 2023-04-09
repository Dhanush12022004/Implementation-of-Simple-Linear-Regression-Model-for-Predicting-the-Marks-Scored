# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for Gradient Design.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given data.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Dhanush.G.R.
RegisterNumber:  212221040038
*/
```
import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error,mean_squared_error

df=pd.read_csv('/content/student_scores.csv')

print('df.head:')

#displaying the content in datafile

df.head()

print("df.tail:")

df.tail()

print("Array value of x : ")

X=df.iloc[:,:-1].values

X

print("Array value of y : ")

Y=df.iloc[:,1].values

Y


#splitting train and test data

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X_train,Y_train)

Y_pred=regressor.predict(X_test)

#displaying predicted values

print("Predicted Values:")

Y_pred

#displaying actual values

print("Actual Values:")

Y_test

#graph plot for training data

print("Graph plot for training data:")

plt.scatter(X_train,Y_train,color="orange")

plt.plot(X_train,regressor.predict(X_train),color="red")

plt.title("Hours vs Scores (Training Set)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.show()

#graph plot for training data

print("Graph plot for training data:")

plt.scatter(X_test,Y_test,color="purple")

plt.plot(X_test,regressor.predict(X_test),color="pink")

plt.title("Hours vs Scores (Training Set)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.show()

print('Values of MSE,MAE,RMSE:')

mse=mean_squared_error(Y_test,Y_pred)

print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)

print('MAE = ',mae)

rmse=np.sqrt(mse)

print("RMSE = " ,rmse)

## Output:

![image](https://user-images.githubusercontent.com/128135558/229563070-bf2c3897-90be-44d5-ac5d-363fc33f497d.png)

![image](https://user-images.githubusercontent.com/128135558/229563222-376077cf-262c-413a-8770-64f4113cf7c6.png)

![image](https://user-images.githubusercontent.com/128135558/229563375-1c9cfe6f-931f-491c-baa2-49d7d07abd37.png)

![image](https://user-images.githubusercontent.com/128135558/229563544-0436778e-1acc-41c7-ad7b-84c36806a548.png)

![image](https://user-images.githubusercontent.com/128135558/230769430-28771c6b-9102-462a-99c4-60371d89eba4.png)

![image](https://user-images.githubusercontent.com/128135558/230769414-690741e5-b93b-4b6b-9f9b-1b4a81a01570.png)

![image](https://user-images.githubusercontent.com/128135558/230769376-213099cd-6829-424c-bbb5-ae779792d792.png)

![image](https://user-images.githubusercontent.com/128135558/230769337-f440d58c-abfa-40d0-abd0-84ce7b7cade2.png)

![image](https://user-images.githubusercontent.com/128135558/229564707-de744960-b596-48fe-8542-ff133685c460.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
