# Deep-Learning-EXP2
**Developing a Neural Network Classification Model**

**AIM**

To develop a neural network classification model for the given dataset.

**THEORY**

The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). 
Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network 
model that can classify a given iris flower into one of these three species based on the provided features.

**Neural Network Model**

<img width="779" height="607" alt="Screenshot 2025-09-21 124902" src="https://github.com/user-attachments/assets/44a0cdfc-69ad-4a44-9fb2-7efd00509346" />


**DESIGN STEPS**

**STEP 1: Import necessary libraries.**

**STEP 2: Load the dataset "customers.csv"**

**STEP 3: Analyse the dataset and drop the rows which has null values.**

**STEP 4: Use encoders and change the string datatypes in the dataset.**

**STEP 5: Calculate correlation matrix ans plot heatmap and analyse the data.**

**STEP 6: Use various visualizations like pairplot,displot,countplot,scatterplot and visualize the data.**

**STEP 7: Split the dataset into training and testing data using train_test_split.**

**STEP 8: Create a neural network model with 2 hidden layers and output layer with four neurons representing multi-classification.**

**STEP 9: Compile and fit the model with the training data**

**STEP 10: Validate the model using training data.**

**STEP 11: Evaluate the model using confusion matrix.**
**PROGRAM**
```
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

customer_df = pd.read_csv('customers.csv')
customer_df.columns

customer_df.dtypes
customer_df.shape

customer_df.isnull().sum()
customer_df_cleaned = customer_df.dropna(axis=0)
customer_df_cleaned.isnull().sum()
customer_df_cleaned.dtypes

customer_df_cleaned['Gender'].unique()
customer_df_cleaned['Ever_Married'].unique()
customer_df_cleaned['Graduated'].unique()

categories_list=[['Male', 'Female'],
           ['No', 'Yes'],
           ['No', 'Yes'],
           ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
            'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
           ['Low', 'Average', 'High']
           ]
enc = OrdinalEncoder(categories=categories_list)
customers_1 = customer_df_cleaned.copy()

customers_1[['Gender',
             'Ever_Married',
              'Graduated','Profession',
              'Spending_Score']] = enc.fit_transform(customers_1[['Gender','Ever_Married','Graduated','Profession','Spending_Score']])

le = LabelEncoder()
customers_1['Segmentation'] = le.fit_transform(customers_1['Segmentation'])
customers_1.dtypes

customers_1 = customers_1.drop('ID',axis=1)
customers_1 = customers_1.drop('Var_1',axis=1)
customers_1.dtypes

# Calculate the correlation matrix
corr = customers_1.corr()

# Plot the heatmap
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="BuPu",
        annot= True)

# Plot scatterplot
plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size',y='Spending_Score',data=customers_1)

customers_1.describe()
customers_1['Segmentation'].unique()

one_hot_enc = OneHotEncoder()
one_hot_enc.fit(y1)
y1.shape

y = one_hot_enc.transform(y1).toarray()
y.shape
y1[0]
y[0]

X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.33,
                                               random_state=50)
X_train[0]
X_train.shape
scaler_age = MinMaxScaler()
scaler_age.fit(X_train[:,2].reshape(-1,1))

X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)

X_train_scaled[:,2] = scaler_age.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_age.transform(X_test[:,2].reshape(-1,1)).reshape(-1)

# Creating the model
model = Sequential([
    Dense(units=8,activation='relu',input_shape=[8]),
    Dense(units=16,activation='relu'),
    Dense(units=4,activation='softmax')
])

model.compile(optimizer='adam',
                 loss= 'categorical_crossentropy',
                 metrics=['accuracy'])

model.fit(x=X_train_scaled,y=y_train,
             epochs= 2000,
             batch_size= 256,
             validation_data=(X_test_scaled,y_test),
             )

metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['loss','val_loss']].plot()
predi = model.predict(X_test_scaled)
predi
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
x_test_predictions.shape
y_test_truevalue = np.argmax(y_test,axis=1)
y_test_truevalue.shape

print(confusion_matrix(y_test_truevalue,x_test_predictions))

print(classification_report(y_test_truevalue,x_test_predictions))

# Saving the Model
model.save('customer_classification_model.h5')

# Saving the data ,PICKLE:  Stores as binary file and then converts
with open('customer_data.pickle', 'wb') as fh:
   pickle.dump([X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le], fh)

# Loading the Model
model = load_model('customer_classification_model.h5')

# Loading the data
with open('customer_data.pickle', 'rb') as fh:
   [X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le]=pickle.load(fh)

x_single_prediction = np.argmax(model.predict(X_test_scaled[1:2,:]), axis=1)

print(x_single_prediction)

print(le.inverse_transform(x_single_prediction))
```
**Name: Moulishwar G**

**Register Number: 2305001020**


**Dataset Information**


<img width="1018" height="291" alt="Screenshot 2025-09-30 091244" src="https://github.com/user-attachments/assets/31bbcc19-fddc-4959-aa0b-e2db584425db" />



**OUTPUT:**

Training Loss,Validation Loss VS Iteration Plot:


<img width="807" height="602" alt="Screenshot 2025-09-21 125312" src="https://github.com/user-attachments/assets/47f13233-a3e3-48a7-8fed-3dccd1ead469" />



**Classification Report**

<img width="611" height="250" alt="Screenshot 2025-09-30 091128" src="https://github.com/user-attachments/assets/081b4af4-c364-4ae8-878a-2fa768400d76" />



**Confusion Matrix**

<img width="311" height="110" alt="Screenshot 2025-09-30 091135" src="https://github.com/user-attachments/assets/6f6a7844-bf18-4ff9-a4a1-054d162af5f2" />




**New Sample Data Prediction**

<img width="1002" height="361" alt="Screenshot 2025-09-30 091146" src="https://github.com/user-attachments/assets/ee289f76-4a40-40fd-8461-75ef27e52d12" />



**RESULT**


Thus a neural network classification model is developed for the given dataset.
