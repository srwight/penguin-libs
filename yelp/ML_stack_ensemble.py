

def stack_predict(models, X):
    '''Making predictions with each model.'''
    predictions = []
    for m in models:
        prediction = m.predict(X)
        # If there are no predictions yet...
        if len(predictions) == 0:
            # the predictions is a list of our one set of predictions...
            predictions = list(map(lambda p:[p], prediction))
        else:
            # if there are, then we append to each prediction.
            for i in range(len(prediction)):
                predictions[i].append(prediction[i])
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