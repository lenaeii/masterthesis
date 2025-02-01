import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from data_reader import read_and_transform

# Example data: Cases and Activities
data = read_and_transform("/Users/bianyingjie/Documents/masterarbeit/data analysis/sepsis.xlsx")
# Step 1: Remove Case 91 from the training data
training_data = [case for case in data if case["case_id"] != "IC"]
test_case = next(case for case in data if case["case_id"] == "IC")  # Store Case 91 separatelyv


# Step 2: Flatten sequences into input-output pairs for training
pairs = []
for case in training_data:
    activities = case["activities"]
    for i in range(len(activities) - 1):
        input_sequence = " ".join(activities[:i+1])  # Convert sequence up to i as a string
        output_activity = activities[i+1]
        pairs.append({"case_id": case["case_id"], "input_sequence": input_sequence, "output_activity": output_activity})

# Convert pairs into a DataFrame
df = pd.DataFrame(pairs)

# Step 3: Encode activities (output)
label_encoder = LabelEncoder()
df["output_encoded"] = label_encoder.fit_transform(df["output_activity"])

# Step 4: Convert input sequences into features
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), binary=True)
X = vectorizer.fit_transform(df["input_sequence"])
y = df["output_encoded"]

# Step 5: Train-Test Split (optional, but useful for evaluating models)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# Step 7: Train k-Nearest Neighbors Model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Step 8: Predict the next activity for Case 91
test_activities = test_case["activities"][:10]
test_input_sequence = " ".join(test_activities)  # Full sequence as input

# Transform the test input sequence
test_input_vectorized = vectorizer.transform([test_input_sequence])

# Predict using Logistic Regression
predicted_next_activity_encoded_lr = lr_model.predict(test_input_vectorized)
predicted_next_activity_lr = label_encoder.inverse_transform(predicted_next_activity_encoded_lr)

# Predict using k-NN
predicted_next_activity_encoded_knn = knn_model.predict(test_input_vectorized)
predicted_next_activity_knn = label_encoder.inverse_transform(predicted_next_activity_encoded_knn)

# Output the results
print("Test Case ID:", test_case["case_id"])
print("Input Sequence:", test_activities)
print("Predicted Next Activity (Logistic Regression):", predicted_next_activity_lr[0])
print("Predicted Next Activity (k-NN):", predicted_next_activity_knn[0])
