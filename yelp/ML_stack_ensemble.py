def stack_predict(models, X):
    '''Making predictions with each model.'''
    from numpy import dstack
    predictions = None
    for m in models:
        prediction = m.predict_proba(X)
        # If there are no predictions yet...
        if predictions is None:
            # the predictions is a list of our one set of predictions...
            predictions = prediction
        else:
            # if there are, then we append to each prediction.
            predictions = dstack((predictions, prediction))
    predictions = predictions.reshape((predictions.shape[0], predictions.shape[1]*predictions.shape[2]))
    return predictions

def fit_stack(models, X, Y):
    '''Fit - Make predictions based on the other models' predictions.'''
    from sklearn.linear_model import LogisticRegression
    stacked_predictions = stack_predict(models, X)
    model = LogisticRegression()
    model.fit(stacked_predictions, Y)
    return model

def stacked_prediction(models, model, X):
    '''Make a prediction with the trained ensemble model.'''
    stacked_prediction = stack_predict(models, X)
    prediction = model.predict(stacked_prediction)
    return prediction
