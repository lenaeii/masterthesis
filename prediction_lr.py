import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Example data: Cases and Activities
data = [
    {"case_id": "101", "activities": ['Start', 'Start', 'toilet', 'toilet', 'sleep', 'sleep', 'toilet', 'toilet', 'sleep', 'sleep', 'toilet', 'toilet', 'relax', 'relax', 'snack', 'snack', 'toilet', 'toilet', 'bathe', 'bathe', 'dress', 'dress', 'personalhygiene', 'personalhygiene', 'toilet', 'toilet', 'outdoors', 'outdoors', 'personalhygiene', 'personalhygiene', 'outdoors', 'outdoors', 'sleep', 'sleep', 'toilet', 'toilet', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'eatingdrinking', 'eatingdrinking', 'cleaning', 'cleaning', 'relax', 'relax', 'personalhygiene', 'personalhygiene', 'relax', 'relax', 'personalhygiene', 'personalhygiene', 'sleep', 'sleep', 'End', 'End']},
    {"case_id": "11", "activities": ['Start', 'Start', 'sleep', 'sleep', 'toilet', 'toilet', 'sleep', 'sleep', 'toilet', 'toilet', 'dress', 'dress', 'sleep', 'sleep', 'toilet', 'toilet', 'mealpreperation', 'mealpreperation', 'eatingdrinking', 'eatingdrinking', 'cleaning', 'cleaning', 'relax', 'relax', 'personalhygiene', 'personalhygiene', 'work', 'work', 'bathe', 'bathe', 'groom', 'groom', 'toilet', 'toilet', 'mealpreperation', 'mealpreperation', 'eatingdrinking', 'eatingdrinking', 'cleaning', 'cleaning', 'read', 'read', 'relax', 'relax', 'toilet', 'toilet', 'personalhygiene', 'personalhygiene', 'snack', 'snack', 'personalhygiene', 'personalhygiene', 'watchtv', 'watchtv', 'outdoors', 'outdoors', 'mealpreperation', 'mealpreperation', 'eatingdrinking', 'eatingdrinking', 'relax', 'relax', 'cleaning', 'cleaning', 'personalhygiene', 'personalhygiene', 'outdoors', 'outdoors', 'snack', 'snack', 'relax', 'relax', 'cleaning', 'cleaning', 'relax', 'relax', 'personalhygiene', 'personalhygiene', 'sleep', 'sleep', 'End', 'End']},
    {"case_id": "131", "activities": ['Start', 'Start', 'toilet', 'toilet', 'sleep', 'sleep', 'medication', 'medication', 'sleep', 'sleep', 'toilet', 'toilet', 'personalhygiene', 'personalhygiene', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'eatingdrinking', 'eatingdrinking', 'cleaning', 'cleaning', 'relax', 'relax', 'personalhygiene', 'personalhygiene', 'groom', 'groom', 'personalhygiene', 'personalhygiene', 'relax', 'relax', 'personalhygiene', 'personalhygiene', 'relax', 'relax', 'snack', 'snack', 'personalhygiene', 'personalhygiene', 'relax', 'relax', 'personalhygiene', 'personalhygiene', 'toilet', 'toilet', 'relax', 'relax', 'personalhygiene', 'personalhygiene', 'outdoors', 'outdoors', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'eatingdrinking', 'eatingdrinking', 'relax', 'relax', 'snack', 'snack', 'read', 'read', 'personalhygiene', 'personalhygiene', 'sleep', 'sleep', 'End', 'End']},
    {"case_id": "141", "activities": ['Start', 'Start', 'toilet', 'toilet', 'sleep', 'sleep', 'toilet', 'toilet', 'sleep', 'sleep', 'toilet', 'toilet', 'sleep', 'sleep', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'personalhygiene', 'personalhygiene', 'dress', 'dress', 'groom', 'groom', 'outdoors', 'outdoors', 'toilet', 'toilet', 'outdoors', 'outdoors', 'relax', 'relax', 'outdoors', 'outdoors', 'toilet', 'toilet', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'eatingdrinking', 'eatingdrinking', 'outdoors', 'outdoors', 'sleep', 'sleep', 'relax', 'relax', 'personalhygiene', 'personalhygiene', 'relax', 'relax', 'snack', 'snack', 'mealpreperation', 'mealpreperation', 'eatingdrinking', 'eatingdrinking', 'cleaning', 'cleaning', 'relax', 'relax', 'snack', 'snack', 'relax', 'relax', 'read', 'read', 'personalhygiene', 'personalhygiene', 'sleep', 'sleep', 'End', 'End']},
    {"case_id": "151", "activities": ['Start', 'Start', 'toilet', 'toilet', 'sleep', 'sleep', 'toilet', 'toilet', 'sleep', 'sleep', 'toilet', 'toilet', 'sleep', 'sleep', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'relax', 'relax', 'toilet', 'toilet', 'work', 'work', 'personalhygiene', 'personalhygiene', 'groom', 'groom', 'personalhygiene', 'personalhygiene', 'snack', 'snack', 'eatingdrinking', 'eatingdrinking', 'cleaning', 'cleaning', 'relax', 'relax', 'snack', 'snack', 'relax', 'relax', 'toilet', 'toilet', 'relax', 'relax', 'snack', 'snack', 'relax', 'relax', 'snack', 'snack', 'eatingdrinking', 'eatingdrinking', 'cleaning', 'cleaning', 'relax', 'relax', 'personalhygiene', 'personalhygiene', 'relax', 'relax', 'outdoors', 'outdoors', 'toilet', 'toilet', 'personalhygiene', 'personalhygiene', 'sleep', 'sleep', 'End', 'End']},
    {"case_id": "161", "activities": ['Start', 'Start', 'toilet', 'toilet', 'sleep', 'sleep', 'toilet', 'toilet', 'sleep', 'sleep', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'eatingdrinking', 'eatingdrinking', 'cleaning', 'cleaning', 'personalhygiene', 'personalhygiene', 'bathe', 'bathe', 'personalhygiene', 'personalhygiene', 'groom', 'groom', 'personalhygiene', 'personalhygiene', 'snack', 'snack', 'toilet', 'toilet', 'relax', 'relax', 'toilet', 'toilet', 'outdoors', 'outdoors', 'snack', 'snack', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'eatingdrinking', 'eatingdrinking', 'cleaning', 'cleaning', 'relax', 'relax', 'snack', 'snack', 'groom', 'groom', 'personalhygiene', 'personalhygiene', 'sleep', 'sleep', 'toilet', 'toilet', 'sleep', 'sleep', 'End', 'End']},
    {"case_id": "171", "activities": ['Start', 'Start', 'toilet', 'toilet', 'sleep', 'sleep', 'personalhygiene', 'personalhygiene', 'groom', 'groom', 'relax', 'relax', 'snack', 'snack', 'relax', 'relax', 'snack', 'snack', 'toilet', 'toilet', 'work', 'work', 'groom', 'groom', 'personalhygiene', 'personalhygiene', 'groom', 'groom', 'personalhygiene', 'personalhygiene', 'toilet', 'toilet', 'snack', 'snack', 'toilet', 'toilet', 'outdoors', 'outdoors', 'relax', 'relax', 'toilet', 'toilet', 'personalhygiene', 'personalhygiene', 'relax', 'relax', 'personalhygiene', 'personalhygiene', 'End', 'End']},
    {"case_id": "201", "activities": ['Start', 'Start', 'toilet', 'toilet', 'sleep', 'sleep', 'toilet', 'toilet', 'sleep', 'sleep', 'toilet', 'toilet', 'personalhygiene', 'personalhygiene', 'outdoors', 'outdoors', 'medication', 'medication', 'bathe', 'bathe', 'dress', 'dress', 'personalhygiene', 'personalhygiene', 'toilet', 'toilet', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'eatingdrinking', 'eatingdrinking', 'cleaning', 'cleaning', 'relax', 'relax', 'toilet', 'toilet', 'relax', 'relax', 'snack', 'snack', 'relax', 'relax', 'personalhygiene', 'personalhygiene', 'toilet', 'toilet', 'outdoors', 'outdoors', 'personalhygiene', 'personalhygiene', 'relax', 'relax', 'watchtv', 'watchtv', 'personalhygiene', 'personalhygiene', 'toilet', 'toilet', 'sleep', 'sleep', 'End', 'End']},
    {"case_id": "21", "activities": ['Start', 'Start', 'toilet', 'toilet', 'sleep', 'sleep', 'snack', 'snack', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'bathe', 'bathe', 'dress', 'dress', 'personalhygiene', 'personalhygiene', 'outdoors', 'outdoors', 'snack', 'snack', 'mealpreperation', 'mealpreperation', 'eatingdrinking', 'eatingdrinking', 'cleaning', 'cleaning', 'work', 'work', 'sleep', 'sleep', 'toilet', 'toilet', 'outdoors', 'outdoors', 'work', 'work', 'snack', 'snack', 'relax', 'relax', 'outdoors', 'outdoors', 'snack', 'snack', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'eatingdrinking', 'eatingdrinking', 'cleaning', 'cleaning', 'watchtv', 'watchtv', 'snack', 'snack', 'toilet', 'toilet', 'snack', 'snack', 'toilet', 'toilet', 'relax', 'relax', 'personalhygiene', 'personalhygiene', 'sleep', 'sleep', 'End', 'End']},
    {"case_id": "211", "activities": ['Start', 'Start', 'toilet', 'toilet', 'sleep', 'sleep', 'toilet', 'toilet', 'dress', 'dress', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'eatingdrinking', 'eatingdrinking', 'relax', 'relax', 'toilet', 'toilet', 'work', 'work', 'snack', 'snack', 'work', 'work', 'personalhygiene', 'personalhygiene', 'work', 'work', 'personalhygiene', 'personalhygiene', 'dress', 'dress', 'personalhygiene', 'personalhygiene', 'outdoors', 'outdoors', 'snack', 'snack', 'relax', 'relax', 'outdoors', 'outdoors', 'relax', 'relax', 'outdoors', 'outdoors', 'personalhygiene', 'personalhygiene', 'snack', 'snack', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'eatingdrinking', 'eatingdrinking', 'cleaning', 'cleaning', 'relax', 'relax', 'outdoors', 'outdoors', 'relax', 'relax', 'personalhygiene', 'personalhygiene', 'groom', 'groom', 'personalhygiene', 'personalhygiene', 'sleep', 'sleep', 'End', 'End']},
    {"case_id": "221", "activities": ['Start', 'Start', 'toilet', 'toilet', 'sleep', 'sleep', 'toilet', 'toilet', 'sleep', 'sleep', 'personalhygiene', 'personalhygiene', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'relax', 'relax', 'work', 'work', 'groom', 'groom', 'personalhygiene', 'personalhygiene', 'groom', 'groom', 'personalhygiene', 'personalhygiene', 'outdoors', 'outdoors', 'relax', 'relax', 'outdoors', 'outdoors', 'relax', 'relax', 'outdoors', 'outdoors', 'relax', 'relax', 'personalhygiene', 'personalhygiene', 'outdoors', 'outdoors', 'sleep', 'sleep', 'relax', 'relax', 'sleep', 'sleep', 'toilet', 'toilet', 'snack', 'snack', 'eatingdrinking', 'eatingdrinking', 'relax', 'relax', 'cleaning', 'cleaning', 'work', 'work', 'groom', 'groom', 'relax', 'relax', 'groom', 'groom', 'personalhygiene', 'personalhygiene', 'sleep', 'sleep', 'End', 'End']},
    {"case_id": "231", "activities": ['Start', 'Start', 'toilet', 'toilet', 'sleep', 'sleep', 'toilet', 'toilet', 'sleep', 'sleep', 'toilet', 'toilet', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'personalhygiene', 'personalhygiene', 'dress', 'dress', 'personalhygiene', 'personalhygiene', 'outdoors', 'outdoors', 'snack', 'snack', 'eatingdrinking', 'eatingdrinking', 'personalhygiene', 'personalhygiene', 'outdoors', 'outdoors', 'relax', 'relax', 'work', 'work', 'personalhygiene', 'personalhygiene', 'snack', 'snack', 'personalhygiene', 'personalhygiene', 'outdoors', 'outdoors', 'mealpreperation', 'mealpreperation', 'eatingdrinking', 'eatingdrinking', 'outdoors', 'outdoors', 'cleaning', 'cleaning', 'outdoors', 'outdoors', 'relax', 'relax', 'outdoors', 'outdoors', 'personalhygiene', 'personalhygiene', 'sleep', 'sleep', 'End', 'End']},
    {"case_id": "241", "activities": ['Start', 'Start', 'toilet', 'toilet', 'sleep', 'sleep', 'toilet', 'toilet', 'sleep', 'sleep', 'mealpreperation', 'mealpreperation', 'eatingdrinking', 'eatingdrinking', 'personalhygiene', 'personalhygiene', 'dress', 'dress', 'personalhygiene', 'personalhygiene', 'groom', 'groom', 'outdoors', 'outdoors', 'toilet', 'toilet', 'snack', 'snack', 'relax', 'relax', 'outdoors', 'outdoors', 'snack', 'snack', 'relax', 'relax', 'cleaning', 'cleaning', 'mealpreperation', 'mealpreperation', 'eatingdrinking', 'eatingdrinking', 'personalhygiene', 'personalhygiene', 'read', 'read', 'relax', 'relax', 'personalhygiene', 'personalhygiene', 'sleep', 'sleep', 'End', 'End']},
    {"case_id": "31", "activities": ['Start', 'Start', 'toilet', 'toilet', 'sleep', 'sleep', 'medication', 'medication', 'sleep', 'sleep', 'toilet', 'toilet', 'mealpreperation', 'mealpreperation', 'relax', 'relax', 'toilet', 'toilet', 'mealpreperation', 'mealpreperation', 'relax', 'relax', 'toilet', 'toilet', 'snack', 'snack', 'eatingdrinking', 'eatingdrinking', 'cleaning', 'cleaning', 'work', 'work', 'relax', 'relax', 'snack', 'snack', 'relax', 'relax', 'personalhygiene', 'personalhygiene', 'relax', 'relax', 'toilet', 'toilet', 'snack', 'snack', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'eatingdrinking', 'eatingdrinking', 'cleaning', 'cleaning', 'watchtv', 'watchtv', 'snack', 'snack', 'watchtv', 'watchtv', 'personalhygiene', 'personalhygiene', 'read', 'read', 'sleep', 'sleep', 'End', 'End']},
    {"case_id": "61", "activities": ['Start', 'Start', 'sleep', 'sleep', 'toilet', 'toilet', 'sleep', 'sleep', 'personalhygiene', 'personalhygiene', 'sleep', 'sleep', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'eatingdrinking', 'eatingdrinking', 'relax', 'relax', 'personalhygiene', 'personalhygiene', 'outdoors', 'outdoors', 'relax', 'relax', 'outdoors', 'outdoors', 'sleep', 'sleep', 'toilet', 'toilet', 'work', 'work', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'eatingdrinking', 'eatingdrinking', 'relax', 'relax', 'cleaning', 'cleaning', 'relax', 'relax', 'personalhygiene', 'personalhygiene', 'groom', 'groom', 'personalhygiene', 'personalhygiene', 'sleep', 'sleep', 'End', 'End']},
    {"case_id": "71", "activities": ['Start', 'Start', 'toilet', 'toilet', 'sleep', 'sleep', 'toilet', 'toilet', 'sleep', 'sleep', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'work', 'work', 'personalhygiene', 'personalhygiene', 'toilet', 'toilet', 'outdoors', 'outdoors', 'toilet', 'toilet', 'relax', 'relax', 'toilet', 'toilet', 'snack', 'snack', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'relax', 'relax', 'cleaning', 'cleaning', 'personalhygiene', 'personalhygiene', 'read', 'read', 'personalhygiene', 'personalhygiene', 'sleep', 'sleep', 'End', 'End']},
    {"case_id": "81", "activities": ['Start', 'Start', 'toilet', 'toilet', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'groom', 'groom', 'bathe', 'bathe', 'dress', 'dress', 'personalhygiene', 'personalhygiene', 'outdoors', 'outdoors', 'snack', 'snack', 'eatingdrinking', 'eatingdrinking', 'personalhygiene', 'personalhygiene', 'outdoors', 'outdoors', 'relax', 'relax', 'snack', 'snack', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'eatingdrinking', 'eatingdrinking', 'personalhygiene', 'personalhygiene', 'relax', 'relax', 'personalhygiene', 'personalhygiene']},


    # Add other cases from your dataset...
    {"case_id": "91", "activities": ['Start', 'Start', 'toilet', 'toilet', 'relax', 'relax', 'mealpreperation', 'mealpreperation', 'groom', 'groom']}
]

# Step 1: Remove Case 91 from the training data
training_data = [case for case in data if case["case_id"] != "91"]
test_case = next(case for case in data if case["case_id"] == "91")  # Store Case 91 separately

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
test_activities = test_case["activities"]
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
