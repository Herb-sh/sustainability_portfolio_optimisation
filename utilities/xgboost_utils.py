import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

def get_xgboost_trained_model(X_train, y_train, X_test, y_test):
    # Cross validation
    # defining model
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror',
        tree_method='hist'
    )
    # defining time-aware CV method - no shuffling
    cv = TimeSeriesSplit(n_splits=5)

    rmse_scorer = make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)))

    # evaluate model with negative MAE (for scikit-learn higher = better)
    scores = cross_val_score(model, X_train, y_train, scoring=rmse_scorer, cv=cv, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
    # convert to positive MAE scores
    mae_scores = np.abs(scores)

    # Report results
    print('Mean MAE: %.3f (%.3f)' % (mae_scores.mean(), mae_scores.std()))
    #
    return model, scores
