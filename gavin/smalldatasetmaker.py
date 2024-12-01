import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
print("Loading dataset...")
df = pd.read_csv("processed_urls.csv")
print("Dataset loaded successfully!")

# Separate features and labels
features = df.columns[:-5]  # All columns except the last 5 are features
labels = df.columns[-5:]    # The last 5 columns are labels
print(f"Identified {len(features)} features and {len(labels)} label columns.")

# Train a Random Forest to calculate feature importance
print("Training Random Forest model...")
clf = RandomForestClassifier(random_state=42)
clf.fit(df[features], df[labels].values.argmax(axis=1))  # Use argmax to simulate a multiclass target
print("Random Forest training completed!")

# Get feature importance and select the top 20%
print("Calculating feature importance...")
importances = pd.Series(clf.feature_importances_, index=features)
top_features = importances.sort_values(ascending=False).head(int(0.2 * len(features))).index
print(f"Selected top {len(top_features)} features based on importance.")

# Filter the dataset to the top features
print("Filtering dataset to top features...")
df_top_features = df[top_features]

# Remove highly correlated features
print("Identifying highly correlated features...")
correlation_matrix = df_top_features.corr(method='spearman')
correlated_pairs = np.where(np.abs(correlation_matrix) > 0.9)  # Threshold for correlation
correlated_pairs = [(correlation_matrix.index[i], correlation_matrix.columns[j]) 
                    for i, j in zip(*correlated_pairs) if i != j and i < j]

# Keep the more important feature of each correlated pair
print("Filtering correlated features...")
features_to_drop = set()
correlation_info = []

for feature1, feature2 in correlated_pairs:
    if importances[feature1] >= importances[feature2]:
        features_to_drop.add(feature2)
        correlation_info.append((feature1, feature2, feature2))
    else:
        features_to_drop.add(feature1)
        correlation_info.append((feature1, feature2, feature1))

# Log the removed correlated features
print("Correlated features removed:")
for feature1, feature2, removed in correlation_info:
    print(f"{feature1} and {feature2} are correlated. Removed: {removed}")

print(f"Dropping {len(features_to_drop)} less important correlated features...")
df_final_features = df_top_features.drop(columns=list(features_to_drop))

# Combine the final features and labels
df_final = pd.concat([df_final_features, df[labels]], axis=1)

# Save the processed dataset
output_file = "processed_top_features_with_labels.csv"
print(f"Saving the processed dataset to {output_file}...")
df_final.to_csv(output_file, index=False)
print("Dataset saved successfully!")
