import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score

st.set_page_config(page_title="Universal ML Tool", layout="wide")

st.title("🚀 Universal Machine Learning Model Selector")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    if target:

        X = df.drop(columns=[target])
        y = df[target]

        # Auto detect problem type
        if y.dtype == "object" or y.nunique() <= 10:
            problem_type = "Classification"
        else:
            problem_type = "Regression"

        st.write(f"### Detected Problem Type: {problem_type}")

        # Identify column types
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = X.select_dtypes(include=["object"]).columns

        # Preprocessing
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols)
            ]
        )

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        train_size = len(X_train)

        # Model Selection
        if problem_type == "Classification":
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "KNN": KNeighborsClassifier(n_neighbors=min(5, train_size))
            }
            scoring = "accuracy"
        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "KNN": KNeighborsRegressor(n_neighbors=min(5, train_size))
            }
            scoring = "r2"

        model_name = st.selectbox("Select Model", list(models.keys()))
        model = models[model_name]

        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        if st.button("Train Model"):

            try:
                # Adaptive CV folds
                if len(X_train) >= 50:
                    cv_folds = 5
                elif len(X_train) >= 20:
                    cv_folds = 3
                else:
                    cv_folds = 2

                scores = cross_val_score(pipe, X_train, y_train, cv=cv_folds, scoring=scoring)

                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)

                st.write("### Cross Validation Score")
                st.write(np.mean(scores))

                if problem_type == "Classification":
                    acc = accuracy_score(y_test, preds)
                    st.write("### Test Accuracy")
                    st.write(acc)
                else:
                    r2 = r2_score(y_test, preds)
                    st.write("### Test R2 Score")
                    st.write(r2)

            except Exception as e:
                st.error(f"Error occurred: {e}")
