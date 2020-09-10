import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
df =pd.read_csv("winequality-red.csv")
"""
print(df)
print(df.shape)
print(df.size)
print(df.info())
print(df.columns)
print(df.describe())
print(df.isnull().sum())
print(df.head())

"""
#converting contionous values into categorical values
values = (1,4,8)  # 1 to 4 are bad && 4 to 8 are good
review = ['bad','good']
df['quality'] = pd.cut(df['quality'],bins=values,labels=review)
#print(df['quality'])
print(df['quality'].value_counts())

#Label encoder converting bad,good  to 0 & 1

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df['quality'] = label.fit_transform(df['quality'])
#print(df['quality'])
print(df['quality'].value_counts())


#X = df[['sulphates']].values
X= df.iloc[:,:11].values
Y = df[['quality']].values
#print(X)
#print(Y)
print(X.size)
print(Y.size)

#Training & Testing data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

"""
print("Xtrain",X_train)
print("Ytrain",Y_train)
print("Xtest",X_test)
print("ytest",Y_test)
print("Xtrain shape",X_train.shape)
print("Ytrain shape",Y_train.shape)
print("Xtest shape",X_test.shape)
print("ytest shape ",Y_test.shape)
print("Xtrain:",X_train.size)
print("Xtest:",X_test.size)
print("Ytrain:",Y_train.size)
print("Ytest:",Y_test.size)
"""

#Logistic Regression

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model = lr.fit(X_train,Y_train)
print(model)

"""
#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
#print(X_train)
#print(X_test)
"""

#Predicting values
ypred = model.predict(X_test)
#print(ypred)

#validation Technique
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,ypred)
print(cm)

#classification report

from sklearn.metrics import classification_report
cr =classification_report(Y_test,ypred)
print(cr)

#Accuracy score for model

from sklearn.metrics import accuracy_score
ac = accuracy_score(Y_test,ypred)*100
print("Accuracy : ",ac)

"""
#Dynamically testing
print(lr.coef_)
print(lr.intercept_)
coefficients = lr.coef_
intercept = lr.intercept_
#print(coefficients[:,0])
#print(intercept)

n1 = float(input("Enter fixed acidity value : "))
n2 = float(input("Entre volatile acidity value : "))
n3 = float(input("Enter citric acid value : "))
n4 = float(input("Enter residual sugar value : "))
n5 = float(input("Enter chlorides value : "))
n6 = float(input("Enter free sulfur dioxide value : "))
n7 = float(input("Enter total sulfur dioxide value : "))
n8 = float(input("Enter density value : "))
n9 = float(input("Enter pH value : "))
n10 = float(input("Enter sulphates value : "))
n11 = float(input("Enter alcohol value : "))


ytesting = intercept + coefficients[:,0]*n1+coefficients[:,1]*n2+coefficients[:,2]*n3+coefficients[:,3]*n4+coefficients[:,4]*n5+coefficients[:,5]*n6+coefficients[:,6]*n7+coefficients[:,7]*n8+coefficients[:,8]*n9+coefficients[:,9]*n10+coefficients[:,10]*n11
print(ytesting)

final = 1.0/(1+np.exp(-ytesting))
print(final)

if final < 0.5:
    print(0)
elif final > 0.5:
    print(1)
"""