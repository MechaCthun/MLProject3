import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

# Input file path
input_file = "processed_top_features_with_labels.csv"  # Replace with your processed file

# Step 1: Load the dataset
df = pd.read_csv(input_file)

# Step 2: Separate features and labels
# Identify one-hot encoded label columns
label_columns = [col for col in df.columns if col.startswith("type_")]
feature_columns = [col for col in df.columns if col not in label_columns]

X = df[feature_columns].values
y = df[label_columns].values.argmax(axis=1)  # Convert one-hot labels to single integer labels
url_types = [col.split("type_")[1] for col in label_columns]  # Extract original URL type names

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify the classes present in y_test
unique_classes = np.unique(y_test)
target_names = [url_types[i] for i in unique_classes]  # Filter target names for present classes

# Step 4: Train the Naïve Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)

# Metrics
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Step 6: Display confusion matrix with actual URL type names
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, index=target_names, columns=target_names)

print("\nConfusion Matrix:")
print(conf_matrix_df)
