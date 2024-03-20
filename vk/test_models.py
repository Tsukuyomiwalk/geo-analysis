import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from joblib import Parallel, delayed
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor


def preprocessing_data(train_df, test_df, features_df):
    step = 2.5
    features_df_copy = features_df.copy()
    features_df_copy['lat_with_step'] = (features_df_copy['lat'] // step) * step
    features_df_copy['lon_with_step'] = (features_df_copy['lon'] // step) * step

    grouped_features_df = features_df_copy.groupby(['lat_with_step', 'lon_with_step']).mean().reset_index()

    for df in [train_df, test_df]:
        df_copy = df.copy()
        df_copy['lat_with_step'] = (df_copy['lat'] // step) * step
        df_copy['lon_with_step'] = (df_copy['lon'] // step) * step
        df_merged = pd.merge(df_copy, grouped_features_df, on=['lat_with_step', 'lon_with_step'], how='left')
        df_merged.drop(['lat_with_step', 'lon_with_step'], axis=1, inplace=True)
        df_merged.reset_index(drop=True, inplace=True)

    return train_df, test_df


def evaluate_model(model, X_train, y_train):
    mae = np.mean(np.abs(y_train - model.predict(X_train)))
    return mae


if __name__ == "__main__":
    train_df = pd.read_csv("datasets/train.csv")
    test_df = pd.read_csv("datasets/test.csv")
    features_df = pd.read_csv("datasets/features.csv")

    train_processed, test_processed = preprocessing_data(train_df, test_df, features_df)

    train_merged = pd.merge(train_df, features_df, on=['lat', 'lon'], how='left')
    test_merged = pd.merge(test_df, features_df, on=['lat', 'lon'], how='left')
    X_train = train_merged.drop(columns=['id', 'score'])
    y_train = train_merged['score']
    X_test = test_merged.drop(columns=['id'])

    selector = VarianceThreshold(threshold=0.05)
    X_train_filtered = selector.fit_transform(X_train)
    X_test_filtered = selector.transform(X_test)

    models = {
        "Random Forest": GridSearchCV(RandomForestRegressor(), param_grid={'n_estimators': [100, 200]},
                                      scoring='neg_mean_absolute_error'),
        "Support Vector Machine": GridSearchCV(SVR(), param_grid={'kernel': ['linear', 'rbf']},
                                               scoring='neg_mean_absolute_error'),
        "Decision Tree": GridSearchCV(DecisionTreeRegressor(),
                                      param_grid={
                                          'criterion': ['absolute_error', 'friedman_mse', 'poisson'],
                                          'max_depth': [None, 5, 10, 15],
                                          'min_samples_leaf': [1, 2],
                                          'max_features': ['sqrt', 'log2']},
                                      scoring='neg_mean_absolute_error'),
        "CatBoost": GridSearchCV(CatBoostRegressor(verbose=0), param_grid={'depth': [4, 6, 8]},
                                 scoring='neg_mean_absolute_error'),
        "Stacking": StackingRegressor(estimators=[
            ('lr', LinearRegression()),
            ('rf', RandomForestRegressor()),
            ('svr', SVR()),
            ('dt', DecisionTreeRegressor())
        ], final_estimator=LinearRegression()),
        "Lasso": GridSearchCV(Lasso(), param_grid={'alpha': [0.1, 0.5]}, scoring='neg_mean_absolute_error')
    }

    best_model = None
    best_mae = float('inf')
    results = Parallel(n_jobs=6)(delayed(model.fit)(X_train_filtered, y_train) for model in models.values())
    maes = Parallel(n_jobs=6)(delayed(evaluate_model)(model, X_train_filtered, y_train) for model in results)
    for name, mae in zip(models.keys(), maes):
        print(f"{name}: MAE = {mae}")
        if mae < best_mae:
            best_mae = mae
            best_model = results[maes.index(mae)]

    pickle.dump(best_model, open('best_model.sav', 'wb'))
    pickle.dump(selector, open('selector.sav', 'wb'))
