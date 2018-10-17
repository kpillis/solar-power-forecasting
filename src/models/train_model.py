from xgboost import XGBRegressor
from sklearn.svm import LinearSVR
import pickle
def train(X,y,boosting):
    if boosting :
        model = XGBRegressor()
    else:
        model = LinearSVR()
    model.fit(X,y)
    pickle.dump(model, open("../../models/model.py","wb"))
    return model