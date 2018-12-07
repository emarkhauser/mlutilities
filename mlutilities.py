# Estimate
def estimate(estimator, parameters, X_train, y_train, X_test):
    from sklearn.model_selection import GridSearchCV
    from sklearn import metrics
    # Fit the model
    model = GridSearchCV(estimator, parameters, cv=5, iid=False, n_jobs=-1)
    model.fit(X_train, y_train)
    print(model.best_score_)                                  
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, model.best_params_[param_name]))
    # Predict using the model
    return model.predict(X_test)
    
# Classification Results
def classificationResults(predicted, y_test):
    print(metrics.classification_report(y_test, predicted))
    print(metrics.confusion_matrix(y_test, predicted))
