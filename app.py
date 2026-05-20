from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


st.set_page_config(page_title="Sales Prediction", layout="centered")
st.title("Sales Prediction")
st.write("Estimate sales from TV, Radio, and Newspaper advertising spend.")


@st.cache_data
def load_data():
    path = Path("input.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource
def train_best_model(df):
    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    results = []
    trained_models = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append(
            {
                "Model": model_name,
                "R2": r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            }
        )
        trained_models[model_name] = model

    results_df = pd.DataFrame(results).sort_values(by="RMSE")
    best_model_name = results_df.iloc[0]["Model"]
    return trained_models[best_model_name], best_model_name, results_df


df = load_data()

if df is None:
    st.warning("Place input.csv in this project folder to train the model and use the app.")
    st.stop()

required_columns = {"TV", "Radio", "Newspaper", "Sales"}
if not required_columns.issubset(df.columns):
    st.error("input.csv must contain TV, Radio, Newspaper, and Sales columns.")
    st.stop()

model, model_name, results_df = train_best_model(df)

st.subheader("Model Performance")
st.dataframe(results_df, use_container_width=True)
st.caption(f"Selected model: {model_name}")

st.subheader("Enter Advertising Spend")
tv = st.number_input("TV", min_value=0.0, value=200.0, step=1.0)
radio = st.number_input("Radio", min_value=0.0, value=30.0, step=1.0)
newspaper = st.number_input("Newspaper", min_value=0.0, value=20.0, step=1.0)

if st.button("Predict Sales"):
    input_data = pd.DataFrame(
        [[tv, radio, newspaper]], columns=["TV", "Radio", "Newspaper"]
    )
    prediction = model.predict(input_data)[0]
    st.metric("Predicted Sales", f"{prediction:.2f}")
