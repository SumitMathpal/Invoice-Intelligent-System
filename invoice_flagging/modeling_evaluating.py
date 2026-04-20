from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    param_grid = {
        "n_estimators": [100],
        "max_depth": [None],
        "min_samples_split": [5],
        "min_samples_leaf": [1],
        "criterion": ["gini"]
    }

    scorer = make_scorer(f1_score, average="binary")

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print("Best Params:", grid_search.best_params_)

    return grid_search


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n{model_name} Performance:")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)

    return acc