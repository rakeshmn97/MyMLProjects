# importing libraries
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
import numpy as np
#importing dataset
data_set= pd.read_csv("DATASET/Wine.csv")
print(data_set.to_string())
#Extracting Independent and dependent Variable
X= data_set.iloc[:, :-1].values
y= data_set.iloc[:, 13].values
df2=pd.DataFrame(X)
print("X=")
print(df2.to_string())
df3=pd.DataFrame(y)
print("Y=")
print(df3.to_string())
# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
from sklearn.linear_model import LogisticRegression
# instantiate the model (using the default parameters)
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
# fit the model with data
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
df2=pd.DataFrame(X_test)
#test data
print(df2.to_string())
#pred. data
print(y_pred)
df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df2.to_string())
#Evaluating the Algorithm
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
