# Ex02-Developing-a-Neural-Network-Classification-ModelAssignment

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.
In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets
You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![WhatsApp Image 2022-09-08 at 1 56 21 PM](https://user-images.githubusercontent.com/89703145/189076869-4d040d99-f470-4fba-acff-4b6377ba9932.jpeg)

### STEP 1:
We start by reading the dataset using pandas.
### STEP 2:
The dataset is then preprocessed, i.e, we remove the features that don't contribute towards the result.
### STEP 3:
The null values are removed aswell
### STEP 4:
The resulting data values are then encoded. We, ensure that all the features are of the type int, or float, for the model to better process the dataset.
### STEP 5:
Once the preprocessing is done, we split the avaliable data into Training and Validation datasets.
### STEP 6:
The Sequential model is then build using 4 dense layers(hidden) and, 1 input and output layer.
### STEP 7:
The model is then complied and trained with the data. A call back method is also implemented to prevent the model from overfitting.
### STEP 8:
Once the model is done training, we validate and use the model to predict values.

## PROGRAM
```python3
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt
```
```python3
data = pd.read_csv("customers.csv")
data.head()
data_cleaned=data.drop(columns=["ID","Var_1"])
data_col=list(data_cleaned.columns)
pd.DataFrame(data_cleaned.isnull().sum())
data_cleaned=data_cleaned.dropna(axis=0)
pd.DataFrame(data_cleaned.isnull().sum())
data_cleaned.shape
data_cleaned.dtypes
```
```python3
data_col_obj=list()
for c in data_col:
  if data_cleaned[c].dtype=='O':
      data_col_obj.append(c)
data_col_obj.remove("Segmentation")
print(data_col_obj)
from sklearn.preprocessing import OrdinalEncoder
data_cleaned[data_col_obj]=OrdinalEncoder().fit_transform(data_cleaned[data_col_obj])
from sklearn.preprocessing import MinMaxScaler
data_cleaned[["Age"]]=MinMaxScaler().fit_transform(data_cleaned[["Age"]])
data_cleaned.head()
y=data_cleaned[["Segmentation"]].values
y=OneHotEncoder().fit_transform(y).toarray()
pd.DataFrame(y)
```
```python3
X=data_cleaned.iloc[:,:-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
corr = data_cleaned.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="BuPu",
        annot= True)
```
```python3
model=Sequential([
    Dense(64,input_shape=X_train.iloc[0].shape,activation="relu"),
    Dense(32,activation='relu'),
    Dense(16,activation='relu'),
    Dense(8,activation='relu'),
    Dense(4,activation='softmax'),
])
model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_accuracy', patience=15)
model.fit(x=X_train,y=y_train,
          epochs=400,
          validation_data=(X_test,y_test),
          verbose=0, 
          callbacks=[early_stop]
          )
metrics = pd.DataFrame(model.history.history)
metrics.head()
```
```python3
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))
plt.subplot(1,2,1)
plt.plot(metrics[['accuracy','val_accuracy']])
plt.legend(["Training Accuracy","Validation Accuracy"])
plt.title("Accuracy vs Test Accuracy")
plt.subplot(1,2,2)
plt.plot(metrics[['loss','val_loss']])
plt.legend(["Training Loss","Validation Loss"])
plt.title("Loss vs Test Loss")
plt.show()
```
```python3
predictions=np.argmax(model.predict(X_test),axis=1)
y_test=np.argmax(y_test, axis=1)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
x_single_prediction = np.argmax(model.predict(X_test.iloc[1:2,:]), axis=1)
display(X_test.iloc[1:2,:])
print("\nThe model has predicted=",x_single_prediction)
```
## Dataset Information

![Screenshot (27)](https://user-images.githubusercontent.com/89703145/189068896-a5bf46b4-2686-48a6-96f5-ed36567c7335.png)

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![Screenshot (26)](https://user-images.githubusercontent.com/89703145/189069013-ef049913-8938-4fdd-9664-5e0c0e022138.png)

### Classification Report

![Screenshot (25)](https://user-images.githubusercontent.com/89703145/189069157-5ecce248-7b4e-4c3a-8fd9-b32788ede9ff.png)

### Confusion Matrix

![Screenshot (24)](https://user-images.githubusercontent.com/89703145/189069264-3fc205c7-5892-460e-b5b6-8fdfd87fec0a.png)

### New Sample Data Prediction

![Screenshot (23)](https://user-images.githubusercontent.com/89703145/189069308-e81ce725-d5a0-4dc3-9b19-191dadf3a4d9.png)

## RESULT
Therefore,hence constructed a Neural Network model for Multiclass Classification.
