# Multiple linear regression
# importing libraries
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
import numpy as np
#importing dataset
data_set= pd.read_csv("DATASET/scrap price.csv")
print(data_set.to_string())
#Extracting Independent and dependent Variable
x= data_set.iloc[:, :-1].values
y= data_set.iloc[:, 25].values
df2=pd.DataFrame(x)
print("X=")
print(df2.to_string())
df3=pd.DataFrame(y)
print("Y=")
print(df3.to_string())
#Catgorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x= LabelEncoder()
x[:, 2]= labelencoder_x.fit_transform(x[:,2])
dt=pd.DataFrame(x)
print("--------------------")
print(dt.to_string())
print("-----------------------")
labelencoder_x= LabelEncoder()
x[:, 3]= labelencoder_x.fit_transform(x[:,3])
dt=pd.DataFrame(x)
print("--------------------")
print(dt.to_string())
print("-----------------------")
labelencoder_x= LabelEncoder()
x[:, 4]= labelencoder_x.fit_transform(x[:,4])
dt=pd.DataFrame(x)
print("--------------------")
print(dt.to_string())
print("-----------------------")
labelencoder_x= LabelEncoder()
x[:, 5]= labelencoder_x.fit_transform(x[:,5])
dt=pd.DataFrame(x)
print("--------------------")
print(dt.to_string())
print("-----------------------")
labelencoder_x= LabelEncoder()
x[:, 6]= labelencoder_x.fit_transform(x[:,6])
dt=pd.DataFrame(x)
print("--------------------")
print(dt.to_string())
print("-----------------------")
labelencoder_x= LabelEncoder()
x[:, 7]= labelencoder_x.fit_transform(x[:,7])
dt=pd.DataFrame(x)
print("--------------------")
print(dt.to_string())
print("-----------------------")
labelencoder_x= LabelEncoder()
x[:, 8]= labelencoder_x.fit_transform(x[:,8])
dt=pd.DataFrame(x)
print("--------------------")
print(dt.to_string())
print("-----------------------")
labelencoder_x= LabelEncoder()
x[:, 14]= labelencoder_x.fit_transform(x[:,14])
dt=pd.DataFrame(x)
print("--------------------")
print(dt.to_string())
print("-----------------------")
labelencoder_x= LabelEncoder()
x[:, 15]= labelencoder_x.fit_transform(x[:,15])
dt=pd.DataFrame(x)
print("--------------------")
print(dt.to_string())
print("-----------------------")
labelencoder_x= LabelEncoder()
x[:, 17]= labelencoder_x.fit_transform(x[:,17])
dt=pd.DataFrame(x)
print("--------------------")
print(dt.to_string())
print("-----------------------")
# Splitting the dataset into training and test set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)
#Fitting the MLR model to the training set:
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train, y_train)
#Predicting the Test set result;
y_pred= regressor.predict(x_test)
#To compare the actual output values for X_test with the predicted value
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df.to_string())
print("Mean")
print(data_set.describe())
print("-------------------------------------")
#Evaluating the Algorithm
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
from sklearn.metrics import r2_score
# predicting the accuracy score
score=r2_score(y_test,y_pred)
print("r2 socre is ",score*100,"%")