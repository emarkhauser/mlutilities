# Estimate
def estimate(estimator, parameters, X_train, y_train, X_test):
    from sklearn.model_selection import GridSearchCV
    # Fit the model
    model = GridSearchCV(estimator, parameters, cv=5, iid=False, n_jobs=-1)
    model.fit(X_train, y_train)
    print("BEST SCORE: %r; BEST PARAMETERS: %s" % (model.best_score_, model.best_params_))
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    params = model.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("Score: %f (%f); Parameters: %r" % (mean, stdev, param))
    print("")
    # Predict using the model
    return model.predict(X_test)
    
# Classification Results
def classificationResults(predicted, y_test):
    from sklearn import metrics
    print("Classification Report")
    print(metrics.classification_report(y_test, predicted))
    print("")
    print("Confusion Matrix")
    print(metrics.confusion_matrix(y_test, predicted))
    print("")
