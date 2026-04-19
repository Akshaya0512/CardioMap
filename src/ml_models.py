from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_models(X_train, X_test, y_train, y_test):

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)

    return {
        "rf_preds": rf_preds,
        "lr_preds": lr_preds
    }
