import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline 
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('D:/house-prices-advanced-regression-techniques/train.csv', sep=',')
X=data.iloc[:,:-1]
y = data[['SalePrice']]
y=np.ravel(y)
X_test=pd.read_csv('D:/house-prices-advanced-regression-techniques/test.csv', sep=',')

A_data=pd.concat([X, X_test],axis=0)
A_data=pd.get_dummies(A_data)
X_test=A_data.iloc[1460:,:]
X_train=A_data.iloc[:1460,:]

pipe=Pipeline([('regressor', XGBRegressor())])
search_space=[{'regressor':[XGBRegressor()], 'regressor__learning_rate': np.arange(0.01, 0.15, 0.02), 'regressor__n_estimators': np.arange(50, 200, 50),
                'regressor__max_depth': np.arange(4, 15, 2)},
              {'regressor':[LGBMRegressor()], 'regressor__learning_rate': np.arange(0.01, 0.15, 0.02), 'regressor__n_estimators': np.arange(50, 200, 50),
                'regressor__max_depth': np.arange(4, 15, 2)},
              {'regressor':[CatBoostRegressor()], 'regressor__learning_rate': np.arange(0.05, 0.13, 0.01)}]

grid=GridSearchCV(pipe, search_space, scoring='neg_root_mean_squared_error', cv=5, verbose=1)
best_model=grid.fit(X_train, y)

#вывод лучшей модели на экран
best_model.best_estimator_.get_params()['regressor']
print(grid.best_params_)

# y_hat=best_model.predict(X_test)

# y_hat.to_csv('D:/house-prices-advanced-regression-techniques/result.csv', sep=',', index=False)


