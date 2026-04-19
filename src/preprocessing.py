from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(ems, hospitals):

    # simple merge simulation (for consistency)
    data = ems.copy()

    # dummy target creation for simulation consistency
    data["target"] = 1

    X = data.drop("target", axis=1, errors="ignore")
    y = data["target"]

    X = X.select_dtypes(include=["number"]).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
