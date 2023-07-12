# Written by Anmol Gulati
# rocks vs mines, logistic regression model, supervised learning data

import numpy as np  # create numpy arrays
import pandas as pd  # loading data into data frames
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score  # find accuracy of our model

# Data collection and data processing

# loading dataset to pandas dataframe
sonar_data = pd.read_csv('sample_data.csv', header=None)
sonar_data.head()

# number of rows and columns
sonar_data.shape

sonar_data.describe()  # describe -> statistical measures of data
sonar_data[60].value_counts()  # prediction will be good as almost equal no. of data entries
sonar_data.groupby(60).mean()

# separating data and labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

# print(X)
# print(Y)

# Training and Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)
# print(X.shape, X_train.shape, X_test.shape) # 21 test data, 187 training data

model = LogisticRegression()  # Model Training -> Logistic Regression
model.fit(X_train, Y_train) # training the logistic regression model with training data

# Model evaluation
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
# print("Accuracy on training data: ", training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy on testing data: ", test_data_accuracy)

# Making a Predictive System
# random sample data of mine from the sonar data
input_data = (0.0366,0.0421,0.0504,0.0250,0.0596,0.0252,0.0958,0.0991,0.1419,0.1847,0.2222,0.2648,0.2508,0.2291,0.1555,0.1863,0.2387,0.3345,0.5233,0.6684,0.7766,0.7928,0.7940,0.9129,0.9498,0.9835,1.0000,0.9471,0.8237,0.6252,0.4181,0.3209,0.2658,0.2196,0.1588,0.0561,0.0948,0.1700,0.1215,0.1282,0.0386,0.1329,0.2331,0.2468,0.1960,0.1985,0.1570,0.0921,0.0549,0.0194,0.0166,0.0132,0.0027,0.0022,0.0059,0.0016,0.0025,0.0017,0.0027,0.0027)
input_data_as_numpy_array = np.asarray(input_data) # changing input_data to numpy array

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 'R':
    print("The object is a Rock")
else:
    print("The object is a Mine")


