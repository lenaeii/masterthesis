import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from data_reader import read_and_transform

# Load and transform the data
data = read_and_transform("/Users/bianyingjie/Documents/masterarbeit/data analysis/sepsis.xlsx")

# Step 1: Remove Case "TE" from the training data
training_data = [case for case in data if case["case_id"] != "D"]
test_case = next(case for case in data if case["case_id"] == "D")  # Store Case "TE" separately

# Step 2: Flatten sequences into input-output pairs for training
pairs = []
for case in training_data:
    activities = case["activities"]
    for i in range(len(activities) - 1):
        input_sequence = " ".join(activities[:i + 1])  # Convert sequence up to i as a string
        output_activity = activities[i + 1]
        pairs.append({"case_id": case["case_id"], "input_sequence": input_sequence, "output_activity": output_activity})

# Convert pairs into a DataFrame
df = pd.DataFrame(pairs)

# Step 4: Encode output activities
label_encoder = LabelEncoder()
df["output_encoded"] = label_encoder.fit_transform(df["output_activity"])

# Step 5: Convert input sequences into features using CountVectorizer
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), binary=True)
X = vectorizer.fit_transform(df["input_sequence"])
y = df["output_encoded"]

# Step 6: Train-Test Split for evaluation (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train K-Nearest Neighbors Classifier
knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)  # Euclidean distance
knn_model.fit(X_train, y_train)

# Step 8: Predict the next activity for the test case
test_activities = test_case["activities"][:3] # Use all except the last activity
test_input_sequence = " ".join(test_activities)  # Full sequence of test case activities

# Transform the test input sequence
test_input_vectorized = vectorizer.transform([test_input_sequence])

# Predict using KNN
predicted_next_activity_encoded_knn = knn_model.predict(test_input_vectorized)
predicted_next_activity_knn = label_encoder.inverse_transform(predicted_next_activity_encoded_knn)

# Output the results
print("Full Test Input Sequence:", test_input_sequence)
print("Predicted Next Activity (K-Nearest Neighbors):", predicted_next_activity_knn[0])
