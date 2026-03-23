
import numpy as np 
import pandas as pd 


data = pd.read_csv('data.csv') 

X = data[['x1' , 'x2' , 'x3']].values 

y = data[['y']].values 

# define weights
W1 = np.array([[1,2,3] , [4,5,6]]) ; 
W2 = np.array([[5,6]]).reshape(1,2)  

# define bias 
b1= np.array([3,2])
b1 = b1.reshape(2,1)   
b2 = np.array([6]).reshape(1,1)  

# sigmoid activation function 
def Activation(x):
    return 1/(1+np.exp(-x)) 


for x in X: 
    x = x.reshape(1,3) 
    
    #layer 1
    Z1 = np.dot(W1, x.T) + b1
    A1 = Activation(Z1)
    #layer 2 
    Z2 = np.dot(W2, A1) + b2 
    A2 = Activation(Z2) 
    
    print(A2) 
    
    
     
     
    









