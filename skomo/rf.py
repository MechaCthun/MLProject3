import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

file_path = 'l1_processed_urls.csv'
data = pd.read_csv(file_path)

z = data[["type_benign","type_defacement","type_malware","type_phishing","type_type"]]
data.drop(columns=["type_benign","type_defacement","type_malware","type_phishing","type_type"], inplace=True)
X = data

y = z.idxmax(axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print results
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))