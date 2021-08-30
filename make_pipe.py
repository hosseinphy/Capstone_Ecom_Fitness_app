
import dill
import numpy as np
import pandas as pd 

import warnings
# sklearn features and models
from sklearn import base
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# errors
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, accuracy_score



class GroupbyEstimator(base.BaseEstimator, base.RegressorMixin):
    
    def __init__(self, column, estimator_factory):
        # column is the value to group by; estimator_factory can be
        # called to produce estimators
        self.column = column
        self.est_fact = estimator_factory
        self.est_dict = {}
    
    def fit(self, X, y):
        X = X.copy()
        X['label'] = y
        # Create an estimator and fit it with the portion in each group
        for key, df_city in X.groupby(self.column):
            self.est_dict[key] = self.est_fact().fit(df_city, df_city['label'])
        return self

    def predict(self, X):
        X = X.copy()
        X['label'] = 0
        predict_dic = {}
        cities = X[self.column].unique().tolist()        

        for key, df_city in X.groupby(self.column):
            predict_dic[key] = self.est_dict[key].predict(df_city)
                                
        ordered_predict_list = [predict_dic[k] for k in cities]
        return np.concatenate(ordered_predict_list)



def category_factory():
    
    min_tree_splits = range(2,6)
    min_tree_leaves = range(2,8)
    nmax_features = range(0,9)
    max_tree_depth = range(0,20)

    # categorical_columns = ['Quarter','Month', 'Week', 'Dayofyear', 'Day']
    categorical_columns = ['Year','Month', 'Week', 'Day','Quarter']
    numeric_columns = ['price', 'ReleaseNumber']
    trans_columns = ColumnTransformer([
        ('numeric', 'passthrough', numeric_columns),
        ('categorical','passthrough' , categorical_columns)

    ])

    features = Pipeline([
        ('columns', trans_columns),
        ('scaler', MaxAbsScaler()),
    ])
    

    param_grid = {
                  'max_depth' : max_tree_depth,
                  'max_features':nmax_features,
                  'min_samples_leaf':min_tree_leaves 
                 }

    gs = GridSearchCV(
                        DecisionTreeRegressor(min_samples_split=2), 
                        param_grid, cv=40, n_jobs=2
                     )

    
    pipe = Pipeline([('feature', features), ('gs_est', gs)])
    
    return pipe




# read hist_data 
hist_data = pd.read_json('hist_data.json')


# daterange
hist_data["Year"] = hist_data.date.dt.year
hist_data["Month"] = hist_data.date.dt.month
hist_data["Week"] = hist_data.date.dt.week
hist_data["Weekday"] = hist_data.date.dt.weekday
hist_data["Day"] = hist_data.date.dt.day
hist_data["Dayofyear"] = hist_data.date.dt.dayofyear
hist_data["Quarter"] = hist_data.date.dt.quarter


df_model = hist_data.drop(columns=['date', 'title', 'sku', 'NewReleaseFlag'])

if __name__ == "__main__":	
	warnings.filterwarnings('ignore')
	print(df_model.head())
	categoty_model =  GroupbyEstimator('product_category', category_factory).fit(df_model, df_model['inventory'])
	with open('cat_model.dill', 'wb') as f:
    		dill.dump(categoty_model, f, recurse=True)

