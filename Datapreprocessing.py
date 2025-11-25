import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# Load datasets
customers = pd.read_csv("customers.csv")
orders = pd.read_csv("orders.csv")
products = pd.read_csv("products.csv")

print("Customer details:\n", customers)
print("\nOrder details:\n", orders)
print("\nProduct details:\n", products)

# Data Cleaning
customers['age'].fillna(customers['age'].mean(), inplace=True)
customers['email'].fillna("N/A", inplace=True)

# Data Integration
merged = customers.merge(orders, on="customer_id").merge(products, on="product_id")

# Data Transformation
merged["total_price"] = merged["quantity"] * merged["price"]
merged["Feed_back"] = np.where(merged["quantity"] > 1, "Good", "Bad")

print("\nCleaned, Integrated, and Transformed Data:\n", merged)

# Encoding
ordinal = OrdinalEncoder()
label = LabelEncoder()

features_encoded = ordinal.fit_transform(merged)
target_encoded = label.fit_transform(merged["Feed_back"])

print("\nFeatures\n", features_encoded)
print("\nTarget\n", target_encoded)
