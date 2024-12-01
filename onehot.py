import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder

# Input and output file paths
input_file = "malicious_phish.csv"  # Replace with your input file
output_file = "processed_urls.csv"  # Replace with your desired output file

# Step 1: Load the dataset
df = pd.read_csv(input_file, header=None, names=["url", "type"])

# Step 2: Extract features from URLs
# Example: Length of URL, number of special characters, domain suffix
df['url_length'] = df['url'].apply(len)
df['num_special_chars'] = df['url'].apply(lambda x: sum(1 for char in x if char in ['?', '&', '/', '=', '.', '-']))
df['num_digits'] = df['url'].apply(lambda x: sum(c.isdigit() for c in x))

# Vectorize the URLs (character n-grams)
vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4), max_features=500)
url_features = vectorizer.fit_transform(df['url']).toarray()
url_feature_names = vectorizer.get_feature_names_out()

# Add the vectorized features to the DataFrame
url_features_df = pd.DataFrame(url_features, columns=[f"url_feature_{name}" for name in url_feature_names])
df = pd.concat([df, url_features_df], axis=1)

# Step 3: One-hot encode the URL type
encoder = OneHotEncoder(sparse_output=False)  # Updated argument name
encoded_labels = encoder.fit_transform(df[['type']])
label_names = encoder.get_feature_names_out(['type'])

# Add encoded labels to the DataFrame
encoded_labels_df = pd.DataFrame(encoded_labels, columns=label_names)
df = pd.concat([df, encoded_labels_df], axis=1)

# Drop the original "url" and "type" columns as they are no longer needed
df.drop(columns=["url", "type"], inplace=True)

# Step 4: Save the processed data to a new CSV file
df.to_csv(output_file, index=False)

print(f"Processed data saved to {output_file}")
