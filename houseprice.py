import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression

from sklearn.datasets import load_boston
boston = load_boston()

data = pd.DataFrame(boston.data)
#print(data.head())
data.columns = boston.feature_names
#print(data.head())

data['PRICE'] = boston.target
print(data.head())
print(data.shape)

X = data.drop(['PRICE'], axis = 1)
y = data['PRICE']



lm = LinearRegression()

lm.fit(X, y)

y_pred = lm.predict(X)


print('MAE:',metrics.mean_absolute_error(y, y_pred))
print('MSE:',metrics.mean_squared_error(y, y_pred))