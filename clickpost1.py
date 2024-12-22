import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
train_df = pd.read_csv("C:/Users/supen/Desktop/New folder/Lead Data Scientist Assignment/train_.csv", engine="python")
pincodes_df = pd.read_csv("C:/Users/supen/Desktop/New folder/Lead Data Scientist Assignment/pincodes.csv", engine="python")
pred_df = pd.read_csv("C:/Users/supen/Desktop/New folder/Lead Data Scientist Assignment/test_.csv", engine="python")
#creating a dictionary of lat and lon with pincodes
pincode_to_lat = pincodes_df.set_index('Pincode')['Latitude'].to_dict()
pincode_to_lon = pincodes_df.set_index('Pincode')['Longitude'].to_dict()
# Function to validate and clean both train_df and pred_df
def data_processing(df, is_train=True):
    # Maping latitude and longitude
    df['pickup_latitude'] = df['pickup_pin_code'].map(pincode_to_lat)
    df['pickup_longitude'] = df['pickup_pin_code'].map(pincode_to_lon)
    df['drop_latitude'] = df['drop_pin_code'].map(pincode_to_lat)
    df['drop_longitude'] = df['drop_pin_code'].map(pincode_to_lon)
    for date_col in ['order_delivered_date', 'order_shipped_date']:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df[f'{date_col}_year'] = df[date_col].dt.year
            df[f'{date_col}_month'] = df[date_col].dt.month
            df[f'{date_col}_day'] = df[date_col].dt.day
            df.drop(columns=[date_col], inplace=True)
        # Drop non-numeric columns
    non_numeric_cols = df.select_dtypes(include=['object']).columns
    df.drop(columns=non_numeric_cols, inplace=True, errors='ignore')
    
    # Fill missing values
    df.fillna(0, inplace=True)
    
    return df

train_df = data_processing(train_df)

pred_df = data_processing(pred_df, is_train=False)

# Ensureing pred_df has the same columns as training database as we are using same model to make prediction on pred database
missing_cols = set(train_df.columns) - set(pred_df.columns)
for col in missing_cols:
    pred_df[col] = 0  # Adding missing columns to pred_df with 0
pred_df = pred_df[train_df.drop(columns=["order_delivery_sla"]).columns]

# Split liting data on bases of input and output for model
y = train_df["order_delivery_sla"]
X = train_df.drop(columns=["order_delivery_sla"])

# spiting in 80 20 for training and tesing purpose
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(random_state=42, n_estimators=30,n_jobs=-1)
rf.fit(X_train, y_train)
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)
rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rfresults = pd.DataFrame(
    {
        "model used": ["random forest"],
        "training mse": [rf_train_mse],
        "training r2": [rf_train_r2],
        "testing mse": [rf_test_mse],
        "testing r2": [rf_test_r2],
    }
)
print(rfresults)
plt.scatter(x=y_train, y=y_rf_train_pred,alpha=0.3)
z=np.polyfit(y_train, y_rf_train_pred, 1)
p=np.poly1d(z)
plt.plot(y_train, p(y_train),'#F8766D')
plt.xlabel("given values")
plt.ylabel("predicted values")
plt.plot()
plt.show()

# Make predictions on the prediction dataset
predictions = rf.predict(pred_df)
submission = pd.DataFrame({'id': pred_df['id'], 'predicted_exact_sla': predictions})
submission.to_csv("C:/Users/supen/Desktop/New folder/Lead Data Scientist Assignment/submission.csv")