import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import time

# Input file path
input_file = "processed_top_features_with_labels.csv"  # Replace with your processed file

print("[INFO] Loading the dataset...")
start_time = time.time()

# Step 1: Load the dataset
df = pd.read_csv(input_file)
print(f"[INFO] Dataset loaded in {time.time() - start_time:.2f} seconds. Shape: {df.shape}")

# Step 2: Separate features and labels
print("[INFO] Separating features and labels...")
label_columns = [col for col in df.columns if col.startswith("type_")]
feature_columns = [col for col in df.columns if col not in label_columns]

X = df[feature_columns].values
y = df[label_columns].values.argmax(axis=1)  # Convert one-hot labels to single integer labels
url_types = [col.split("type_")[1] for col in label_columns]  # Extract original URL type names
print(f"[INFO] Features shape: {X.shape}, Labels shape: {y.shape}")

# Step 3: Split the data into training and testing sets
print("[INFO] Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"[INFO] Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

# Identify the classes present in y_test
unique_classes = np.unique(y_test)
target_names = [url_types[i] for i in unique_classes]  # Filter target names for present classes
print(f"[INFO] Classes present in the dataset: {target_names}")

# Step 4: Train the Random Forest model
print("[INFO] Training the Random Forest model...")
start_time = time.time()
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)  # Use all available cores
model.fit(X_train, y_train)
print(f"[INFO] Random Forest model trained in {time.time() - start_time:.2f} seconds.")

# Step 5: Evaluate the model
print("[INFO] Evaluating the model on the test set...")
y_pred = model.predict(X_test)

# Metrics
print("[INFO] Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

accuracy = accuracy_score(y_test, y_pred)
print(f"[INFO] Accuracy: {accuracy:.2f}")

# Step 6: Display confusion matrix with actual URL type names
print("[INFO] Generating confusion matrix...")
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, index=target_names, columns=target_names)

print("\nConfusion Matrix:")
print(conf_matrix_df)

print("[INFO] Script completed successfully.")
