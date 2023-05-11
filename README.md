# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the data file and import numpy, matplotlib and scipy.
2. Visulaize the data and define the sigmoid function, cost function and gradient descent.
3. Plot the decision boundary .
4. Calculate the y-prediction.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by : YOHESH KUMAR R.M
RegisterNumber:  212222240118
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize    #to remove unwanted data and memory storage

data=np.loadtxt("/content/ex2data1 (1).txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

Visualizing the data
plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

Sigmoid fuction
def sigmoid(z):
  return 1/(1+np.exp(-z))
  
plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFuction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J= -(np.dot(y, np.log(h)) + np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T, h-y) / X.shape[0]
  return J,grad
  
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J, grad=costFuction(theta, X_train, y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J, grad=costFuction(theta, X_train, y)
print(J)
print(grad)

def cost(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  J= -(np.dot(y, np.log(h)) + np.dot(1-y, np.log(1-h))) / X.shape[0]
  return J
  
def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad= np.dot(X.T, h-y) / X.shape[0]
  return grad
  
X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta= np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train,y),method="Newton-CG",jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min, x_max = X[:,0].min() - 1, X[:,0].max()+1
  y_min, y_max = X[:,1].min() - 1, X[:,1].max()+1
  xx, yy = np.meshgrid(np.arange(x_min,x_max,0.1),
                       np.arange(y_min,y_max,0.1))
  X_plot = np.c_[xx.ravel(), yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot, theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not admitted")
  plt.contour(xx,yy,y_plot, levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
  plotDecisionBoundary(res.x,X,y)
  
prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob = sigmoid(np.dot(X_train,theta))
  return(prob >= 0.5).astype(int)
  
np.mean(predict(res.x,X)==y)
```


## Output:
### Array Value of x :
![av x](https://github.com/yoheshkumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393568/af65a5e1-de70-450d-87f8-d709f78ef7cc)
### Array Value of y :
![av y](https://github.com/yoheshkumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393568/44b21004-cdc3-4b7d-a8de-7a43605f4bfd)
### Exam 1 - Score Graph :
![ex 1](https://github.com/yoheshkumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393568/ba8261c5-e08b-4733-b090-aaeee04fe317)
### Sigmoid Function Graph :
![sig fun](https://github.com/yoheshkumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393568/6c46bbc0-b443-4a93-b082-315f2b1888eb)
### X_train_grad Value :
![av y](https://github.com/yoheshkumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393568/44b21004-cdc3-4b7d-a8de-7a43605f4bfd)
### Y_train_grad Value :
![y train](https://github.com/yoheshkumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393568/c34541f4-e817-4436-acba-7cd7ad4650c8)
### Print Res.x :
![pvm](https://github.com/yoheshkumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393568/119fd9c2-1a6c-433f-bb92-2ec84461c71a)
### Decision Boundary - Graph For Exam Score :
![db](https://github.com/yoheshkumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393568/a0660774-ea0d-496b-8f16-5903de8b52a8)
### Proability Value :
![pv](https://github.com/yoheshkumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393568/0320e405-e5e5-4339-b871-3f0c7cd7b73f)
### Prediction Value of Mean :
![pvm](https://github.com/yoheshkumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393568/119fd9c2-1a6c-433f-bb92-2ec84461c71a)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

