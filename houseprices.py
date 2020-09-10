import numpy as np
import pandas as pd
df = pd.read_csv("USA_Housing.csv")
""""
print(df)
print(df.shape)
print(df.size)
print(df.columns) # Displaying All column names
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.head())
print(df.info())
"""
#X = df.iloc[:,[0,1,2,3,4]].values
X = df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]
Y = df['Price'].values
#Y = df.iloc[:,5].values
y = np.mean(Y)
"""
print(X)
print(Y)
print(X.shape)
print(Y.shape)
print(X.size)
print(Y.size)
"""

# splitting Data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.25,random_state = 0)
"""
print(X_train)
print((X_test))
print(Y_train)
print(Y_test)
print("Xtrain:",X_train.size)
print("Xtest:",X_test.size)
print("Ytrain:",Y_train.size)
print("Ytest:",Y_test.size)
"""

#importing Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
print(model.fit(X_train,Y_train))
print("Co-efficients : ",model.coef_)
print("Intercept : ",model.intercept_)

# predicting y values

ypredict = model.predict(X_test)
print("Predicted values : ",ypredict)


#Root Mean squared error
from sklearn.metrics import mean_squared_error
print("Training score : ",model.score(X_train,Y_train))
print("Testing score : ",model.score(X_test,Y_test))

#importing r2 score & calculating R-Squared value

from sklearn.metrics import r2_score
print("R2 score :",r2_score(Y_test,ypredict))

"""
rsquare = sum((ypredict-y)**2)/sum((Y_test-y)**2)
print("r",rsquare)


#calculating R-Square method in python manually
def get_r2_numpy():
    slope, intercept = np.polyfit(Y_test, ypredict, 1)
    r_squared = 1 - (sum((ypredict - (slope * Y_test + intercept)) ** 2) / ((len(ypredict) - 1) * np.var(ypredict, ddof=1)))
    print(r_squared)
get_r2_numpy()


#Another way to calculate R-sqaured method
def compute_r2():
    sse = sum((Y_test - ypredict) ** 2)
    tse = (len(Y_test) - 1) * np.var(Y_test, ddof=1)
    r2_score = 1 - (sse / tse)
    print(r2_score)
compute_r2()
"""
