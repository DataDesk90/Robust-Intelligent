import streamlit as st
import pandas as pd
import numpy as np
pandas
numpy
scikit-learn
matplotlib
seaborn

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from collections import Counter

st.set_page_config(page_title="Robust AutoML Tool", layout="wide")
st.title("🚀 Robust Intelligent AutoML System")

uploaded_file = st.file_uploader("Upload ANY CSV Dataset", type=["csv"])

if uploaded_file is not None:

    # -----------------------
    # SAFE CSV LOADING
    # -----------------------
    try:
        df = pd.read_csv(uploaded_file)
        if df.shape[1] == 1:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';')
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=';')

    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)

    df = df.drop_duplicates()

    target = st.selectbox("Select Target Column", df.columns)

    if target:

        X = df.drop(columns=[target])
        y = df[target]

        # -----------------------
        # PROBLEM TYPE DETECTION
        # -----------------------
        if y.dtype == "object" or len(np.unique(y)) <= 15:
            problem_type = "Classification"
            st.success("Detected: Classification Problem")
        else:
            problem_type = "Regression"
            st.success("Detected: Regression Problem")

        # -----------------------
        # FEATURE TYPES
        # -----------------------
        numeric_features = X.select_dtypes(include=np.number).columns
        categorical_features = X.select_dtypes(exclude=np.number).columns

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ])

        if problem_type == "Classification":
            le = LabelEncoder()
            y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # -----------------------
        # MODEL SET
        # -----------------------
        if problem_type == "Classification":
            models = {
                "Logistic Regression": LogisticRegression(max_iter=2000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "SVM": SVC(),
                "KNN": KNeighborsClassifier()
            }
            scoring = "accuracy"
        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "SVR": SVR(),
                "KNN Regressor": KNeighborsRegressor()
            }
            scoring = "r2"

        results = {}

        for name, model in models.items():

            pipe = Pipeline([
                ("preprocessing", preprocessor),
                ("model", model)
            ])

            try:
                # -----------------------
                # ADAPTIVE CV
                # -----------------------
                if problem_type == "Classification":
                    class_counts = Counter(y_train)
                    min_class = min(class_counts.values())

                    if min_class >= 5:
                        cv_folds = 5
                    elif min_class >= 3:
                        cv_folds = 3
                    else:
                        cv_folds = 2
                else:
                    if len(X_train) >= 50:
                        cv_folds = 5
                    elif len(X_train) >= 20:
                        cv_folds = 3
                    else:
                        cv_folds = 2

                scores = cross_val_score(pipe, X_train, y_train,
                                         cv=cv_folds,
                                         scoring=scoring)

                results[name] = scores.mean()

            except:
                # FALLBACK if CV fails
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)

                if problem_type == "Classification":
                    results[name] = accuracy_score(y_test, preds)
                else:
                    results[name] = r2_score(y_test, preds)

        st.subheader("Model Performance")

        for name, score in results.items():
            st.write(f"{name}: {round(score,4)}")

        best_model_name = max(results, key=results.get)
        st.success(f"🏆 Best Model Selected: {best_model_name}")

        final_model = Pipeline([
            ("preprocessing", preprocessor),
            ("model", models[best_model_name])
        ])

        final_model.fit(X_train, y_train)
        preds = final_model.predict(X_test)

        if problem_type == "Classification":
            st.write("Final Accuracy:", round(accuracy_score(y_test, preds),4))
        else:
            st.write("Final R2:", round(r2_score(y_test, preds),4))

            st.write("Final MSE:", round(mean_squared_error(y_test, preds),4))

