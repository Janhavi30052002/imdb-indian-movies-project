import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def build_preprocessor():
    categorical = ["genre", "director"]
    numeric = ["year", "actor_experience", "director_experience"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric)
        ]
    )

    return pre


def train_model(df):
    X = df[["genre", "director", "year", "actor_experience", "director_experience"]]
    y = df["imdb_rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05
    )

    pre = build_preprocessor()

    pipe = Pipeline([
        ("pre", pre),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    return pipe, X_test, y_test
