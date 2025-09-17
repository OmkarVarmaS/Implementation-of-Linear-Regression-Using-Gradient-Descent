# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe. 
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Omkar Varma S
RegisterNumber:  212224240108

import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  


X = X.flatten()
y = y.flatten()


m = 0  
b = 0  


learning_rate = 0.01
epochs = 1000
n = len(X)


for i in range(epochs):
    y_pred = m * X + b
    error = y_pred - y

   
    dm = (2/n) * np.dot(error, X)
    db = (2/n) * np.sum(error)

   
    m -= learning_rate * dm
    b -= learning_rate * db

    
    if i % 100 == 0:
        cost = np.mean(error ** 2)
        print(f"Epoch {i}: Cost = {cost:.4f}, m = {m:.4f}, b = {b:.4f}")


print(f"\nFinal Model: y = {m:.2f}x + {b:.2f}")

plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, m * X + b, color='red', label='Regression line')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()

*/

```

## Output:


<img width="819" height="781" alt="image" src="https://github.com/user-attachments/assets/25ed4a81-73fa-41f4-af4a-2c09de5f7e61" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
