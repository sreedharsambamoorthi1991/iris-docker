# model_train.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

os.makedirs("artifacts", exist_ok=True)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
print(f"Test accuracy: {acc:.4f}")

joblib.dump(clf, "artifacts/model.joblib")
print("Saved model to artifacts/model.joblib")
