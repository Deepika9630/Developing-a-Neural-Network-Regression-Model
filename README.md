# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:DEEPIKA R

### Register Number:212224230054

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("C:/Users/admin/Desktop/dl/.venv/exp1.csv")
df
x=df[["Input"]].values
y=df[["Output"]].values
xt,xst,yt,yst=train_test_split(x,y,test_size=0.2,random_state=52)
scale1=MinMaxScaler()
xt=scale1.fit_transform(xt)
xst=scale1.transform(xst)
xt1=torch.FloatTensor(xt)
xst1=torch.FloatTensor(xst)
yt1=torch.FloatTensor(yt)
yst1=torch.FloatTensor(yst)
class neuralnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(1,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1))
    def forward(self,x):
        return self.network(x)
model=neuralnet()
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)
epochs=1000
losses=[]
for i in range(epochs):
    optimizer.zero_grad()
    pred=model(xt1)
    loss=criterion(pred,yt1)
    loss.backward()
    optimizer.step()

    if(i%50==0):
        print(f"{i}/{epochs}:",loss.item())   
        losses.append(loss.item())
x11=np.array([[16]])
ip=torch.FloatTensor(scale1.transform(x11))
pred=model(ip)
print(pred.item())
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

```

### Dataset Information
<img width="1062" height="81" alt="image" src="https://github.com/user-attachments/assets/ad70f2b2-833c-454d-90f5-5340b393ab58" />
<img width="165" height="518" alt="image" src="https://github.com/user-attachments/assets/79e82993-8392-46b7-8784-35003aed16e9" />


### OUTPUT

### Training Loss Vs Iteration Plot
<img width="1070" height="648" alt="image" src="https://github.com/user-attachments/assets/85cc4173-0c6b-454c-9bd8-2c8082a78c6c" />

### New Sample Data Prediction
Include your sample input and output here
<img width="1080" height="812" alt="image" src="https://github.com/user-attachments/assets/b3819d6b-1fdd-46c4-9ce5-2fc38e8dc605" />
<img width="1061" height="141" alt="image" src="https://github.com/user-attachments/assets/98a13361-3559-4386-95eb-b4a3c11f61bf" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
