def predict(X_test, model, keep_negative=False):
     
    y_pred = model.predict(X_test)
    if not keep_negative:
        y_pred[y_pred < 0] = 0
    
    return y_pred