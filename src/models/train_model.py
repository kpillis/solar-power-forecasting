from xgboost import XGBRegressor
def train(X,y,linear=False):
    if linear :
        model = XGBRegressor(booster="gblinear")
    else:
        model = XGBRegressor()
    model.fit(X,y)
    return model