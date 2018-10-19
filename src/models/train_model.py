from xgboost import XGBRegressor
def train(X,y,tuned=False):
    if not tuned :
        model = XGBRegressor()
    else:
        model = XGBRegressor(colsample_bytree=0.9,gamma=0.65,learning_rate= 0.5,max_depth=1, min_child= 9.0, n_estimators = 45, subsample = 0.9474048013202294)
    model.fit(X,y)
    return model