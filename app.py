import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import streamlit as st

# Load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    # Drop missing values
    df.dropna(inplace=True)

    # Convert data types
    df['floor_area_sqm'] = df['floor_area_sqm'].astype(float)

    # Feature engineering
    df['year'] = pd.to_datetime(df['month']).dt.year
    df['lease_remaining'] = 99 - (df['year'] - df['lease_commence_date'])
    
    # Additional feature engineering
    df['storey_range'] = df['storey_range'].apply(lambda x: int(x.split(' ')[0]))

    return df

# Load the dataset
file_path = 'resale_flat_prices.csv'
df = load_data(file_path)
df = preprocess_data(df)

# Save preprocessed data (optional)
df.to_csv('preprocessed_resale_flat_prices.csv', index=False)

# Model training
def train_model(df):
    # Select features and target
    X = df[['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'flat_model', 'lease_commence_date', 'lease_remaining']]
    y = df['resale_price']

    # Convert categorical features to numerical
    X = pd.get_dummies(X, columns=['town', 'flat_type', 'flat_model'])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2 Score:", r2_score(y_test, y_pred))

    # Save the model and columns
    joblib.dump(model, 'resale_price_model.pkl')
    joblib.dump(X.columns, 'model_columns.pkl')
    
    return model, X.columns

# Train the model
model, model_columns = train_model(df)

# Streamlit web application
def main():
    st.title("Singapore Resale Flat Price Prediction")

    # User inputs
    town = st.selectbox("Town", df['town'].unique())
    flat_type = st.selectbox("Flat Type", df['flat_type'].unique())
    storey_range = st.slider("Storey Range", min_value=1, max_value=50)
    floor_area_sqm = st.number_input("Floor Area (sqm)")
    flat_model = st.selectbox("Flat Model", df['flat_model'].unique())
    lease_commence_date = st.number_input("Lease Commence Date", min_value=1960, max_value=2024)
    lease_remaining = 99 - (2024 - lease_commence_date)

    user_input = {
        "town": town,
        "flat_type": flat_type,
        "storey_range": storey_range,
        "floor_area_sqm": floor_area_sqm,
        "flat_model": flat_model,
        "lease_commence_date": lease_commence_date,
        "lease_remaining": lease_remaining
    }

    # Convert user input to DataFrame
    user_df = pd.DataFrame(user_input, index=[0])
    user_df = pd.get_dummies(user_df)
    user_df = user_df.reindex(columns=model_columns, fill_value=0)

    if st.button("Predict Resale Price"):
        prediction = model.predict(user_df)
        st.write(f"Predicted Resale Price: ${prediction[0]:,.2f}")

if __name__ == "__main__":
    main()
