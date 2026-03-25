
import pandas as pd 
import numpy as np 

from sklearn.preprocessing import StandardScaler  
 
df = pd.read_csv('lpa.csv') 



X = df[['x1' ,  'x2']].values  

y = df[['y']].values.reshape(-1,1) 

## Normalize 

scaler = StandardScaler() 

X_scaled = scaler.fit_transform(X) 


## Initialize parameters 

row = X_scaled.shape[0] 

col = X_scaled.shape[1]


w = np.ones((col , 1)) ## weight in the basis of features 

b = 0 

learning_rate = 0.01 

epochs = 100 

### Batch Gradient Descent 

for i in range(epochs): 
    y_pred = np.dot(X_scaled , w) + b  ## i dont reshape X_scaled means 
    #####################################i dont do the X as column vector
    
    error = y-y_pred 
    
    loss = 1/row * np.sum((error)**2) 
    
    
    
    ##Gradients
    dw = -2/row * np.dot(X_scaled.T , error)
    db = -2/row * np.sum(error) 
    
    ##Update 
    w = w - learning_rate * dw 
    b = b - learning_rate * db 
    
    if i%2==0:
        print(f"Epoch {i} , loss={loss}")


    
    
    

















