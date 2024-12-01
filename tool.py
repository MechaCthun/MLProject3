import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from urllib.parse import urlparse
import numpy as np
import time

# Input file path
input_file = "processed_top_features_with_labels.csv"  # Replace with your processed file path

# Step 1: Load the dataset
print("[INFO] Loading the dataset...")
start_time = time.time()
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
unique_classes = np.unique(y_test)  # Get unique classes in y_test
filtered_target_names = [url_types[i] for i in unique_classes]  # Map unique classes to URL type names

print("[INFO] Classification Report:")
print(classification_report(y_test, y_pred, labels=unique_classes, target_names=filtered_target_names))

accuracy = accuracy_score(y_test, y_pred)
print(f"[INFO] Accuracy: {accuracy:.2f}")

# Step 6: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=unique_classes)
conf_matrix_df = pd.DataFrame(conf_matrix, index=filtered_target_names, columns=filtered_target_names)

print("\nConfusion Matrix:")
print(conf_matrix_df)

# Step 7: Real-time URL prediction
print("\n[INFO] Model is ready for real-time URL classification!")
print("Type a URL to classify or type 'exit' to quit.")

# Preprocessing function
def preprocess_url(input_url, feature_columns):
    """
    Convert an input URL into a feature vector matching the feature columns.
    """
    # Parse the URL
    parsed_url = urlparse(input_url)
    url_length = len(input_url)
    num_digits = sum(c.isdigit() for c in input_url)
    num_special_chars = sum(not c.isalnum() for c in input_url)
    
    # Feature engineering based on keywords
    url_features = {
        "num_special_chars": num_special_chars,
        "url_length": url_length,
        "url_feature_//": "//" in input_url,
        "url_feature_www.": "www." in input_url,
        "url_feature_/www": "/www" in input_url,
        "num_digits": num_digits,
        "url_feature_m/": "/m" in input_url,
        "url_feature_tt": "tt" in input_url,
        "url_feature_ptio": "ptio" in input_url,
        "url_feature_x.p": "x.p" in input_url,
        "url_feature_/w": "/w" in input_url,
        "url_feature_.com": ".com" in input_url,
        "url_feature_om": "om" in input_url,
        "url_feature_dex": "dex" in input_url,
        "url_feature_co": "co" in input_url,
        "url_feature_ht": "ht" in input_url,
        "url_feature_.c": ".c" in input_url,
        "url_feature_=com": "=com" in input_url,
        "url_feature_.i": ".i" in input_url,
        "url_feature_ml": "ml" in input_url,
        "url_feature_logi": "logi" in input_url,
        "url_feature_.e": ".e" in input_url,
        "url_feature_.htm": ".htm" in input_url,
        "url_feature_s/": "s/" in input_url,
        "url_feature_or": "or" in input_url,
        "url_feature_s-": "s-" in input_url,
        "url_feature_in": "in" in input_url,
        "url_feature_gin": "gin" in input_url,
        "url_feature_ne": "ne" in input_url,
        "url_feature_en": "en" in input_url,
        "url_feature_et": "et" in input_url,
        "url_feature_ent&": "ent&" in input_url,
        "url_feature_re": "re" in input_url,
        "url_feature_de": "de" in input_url,
        "url_feature_-t": "-t" in input_url,
        "url_feature_php": "php" in input_url,
        "url_feature_pa": "pa" in input_url,
        "url_feature_u/": "u/" in input_url,
        "url_feature_tm": "tm" in input_url,
        "url_feature_er": "er" in input_url,
        "url_feature_ex": "ex" in input_url,
        "url_feature_l/": "l/" in input_url,
        "url_feature_.m": ".m" in input_url,
        "url_feature_at": "at" in input_url,
        "url_feature_a/": "a/" in input_url,
        "url_feature_al": "al" in input_url,
        "url_feature_log": "log" in input_url,
        "url_feature_og": "og" in input_url,
        "url_feature_es": "es" in input_url,
        "url_feature_e-": "e-" in input_url,
        "url_feature_t/": "t/" in input_url,
    }

    # Map features to the training dataset
    feature_array = []
    for feature in feature_columns:
        feature_array.append(int(url_features.get(feature, 0)))  # 0 if feature is not in our mapping

    return np.array(feature_array)

# Real-time classification loop
while True:
    user_input = input("\nEnter a URL: ").strip()
    
    if user_input.lower() == "exit":
        print("[INFO] Exiting the script. Goodbye!")
        break

    print("[INFO] Processing the input URL...")
    input_features = preprocess_url(user_input, feature_columns)
    
    if input_features is None or len(input_features) != len(feature_columns):
        print("[ERROR] Could not preprocess the input URL properly.")
        continue
    
    input_features = input_features.reshape(1, -1)  # Reshape for model prediction
    prediction = model.predict(input_features)
    predicted_label = url_types[prediction[0]]
    
    print(f"[INFO] The URL is classified as: {predicted_label}")
