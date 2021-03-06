import pandas as pd
import sklearn
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet

if __name__ == "__main__":
    dataset = pd.read_csv('./data/MyEBirdData.csv')
    print(dataset.describe())

    X = dataset[['Duration (Min)']]
    y = dataset[['Number of Observers']]
    
    print(X)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

    modelLinear = LinearRegression().fit(X_train, y_train)
    y_predict_linear =  modelLinear.predict(X_test)
    plt.hist(y_predict_linear)
    plt.show()
    
    modelLasso = Lasso(alpha=0.02).fit(X_train, y_train)
    y_predict_lasso = modelLasso.predict(X_test)

    modelRidge = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_ridge = modelRidge.predict(X_test)

    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print("Linear Loss:", linear_loss)

    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print("Lasso Loss: ", lasso_loss)

    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print("Ridge Loss: ", ridge_loss)

    print("="*32)
    print("Coef LASSO")
    print(modelLasso.coef_)
    
    print("="*32)
    print("Coef RIDGE")
    print(modelRidge.coef_)

#implementacion_lasso_ridge