import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# === LOAD OBJECT-CENTRIC EVENT DATA ===
file_path = '/Users/bianyingjie/Documents/masterarbeit/data analysis/ocel2-export.json'
with open(file_path, 'r') as file:
    raw_data = json.load(file)

# Extract relevant information and create the DataFrame
events = raw_data['events']
objects = raw_data['objects']

# Create a mapping from object ID to its type
object_type_map = {obj['id']: obj['type'] for obj in objects}

# Create a DataFrame to store event sequences
data = []
for event in events:
    event_id = event['id']
    event_type = event['type']
    timestamp = event['time']
    related_objects = [rel['objectId'] for rel in event['relationships']]
    for obj_id in related_objects:
        case_id = obj_id
        object_type = object_type_map.get(obj_id, 'unknown')
        data.append({
            'case_id': case_id,
            'event_id': event_id,
            'event_type': event_type,
            'timestamp': timestamp,
            'object_type': object_type
        })

df = pd.DataFrame(data)
df = df.sort_values(by=['case_id', 'timestamp'])

# Create sequences of events for each case
df['event_sequence'] = df.groupby('case_id')['event_type'].transform(lambda x: ';'.join(x))

# Ensure event_sequence has at least 2 elements
df = df[df['event_sequence'].apply(lambda x: len(x.split(';')) > 1)]

# Split event_sequence into previous_sequence and next_activity
df['previous_sequence'] = df['event_sequence'].apply(lambda x: ';'.join(x.split(';')[:-1]))
df['next_activity'] = df['event_sequence'].apply(lambda x: x.split(';')[-1])

# Remove rows with missing values
df = df.dropna(subset=['previous_sequence', 'next_activity'])

print(df.tail())

# === ENCODE OUTPUT ACTIVITIES ===
label_encoder = LabelEncoder()
df["output_encoded"] = label_encoder.fit_transform(df["next_activity"])

# === VECTORIZE INPUT SEQUENCES ===
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(';'), binary=True)
X = vectorizer.fit_transform(df["previous_sequence"])
y = df["output_encoded"]

# Ensure X and y have the same number of samples
min_length = min(X.shape[0], len(y))
X, y = X[:min_length], y[:min_length]

# === TRAIN-TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === TRAIN MODELS ===

# Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Neural Network Classifier
nn_model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
nn_model.fit(X_train, y_train)

# === PREDICT NEXT ACTIVITY FOR LAST LOG ENTRY ===
last_case_id = df.iloc[-1]["case_id"]
last_sequence = df.iloc[-1]["previous_sequence"]  # Sequence before last event
last_object_type = df.iloc[-1]["object_type"]  # Extract associated object type
actual_next_activity = df.iloc[-1]["next_activity"]  # Ground truth

test_input_vectorized = vectorizer.transform([last_sequence])

# Predictions
predicted_next_activity_dt = label_encoder.inverse_transform(dt_model.predict(test_input_vectorized))[0]
predicted_next_activity_rf = label_encoder.inverse_transform(rf_model.predict(test_input_vectorized))[0]
predicted_next_activity_nn = label_encoder.inverse_transform(nn_model.predict(test_input_vectorized))[0]

# === CHECK IF PREDICTIONS ARE CORRECT ===
correct_dt = predicted_next_activity_dt == actual_next_activity
correct_rf = predicted_next_activity_rf == actual_next_activity
correct_nn = predicted_next_activity_nn == actual_next_activity

# === PRINT PREDICTIONS ===
print("\n=== NEXT ACTIVITY PREDICTIONS ===")
print(f"Case ID: {last_case_id}")
print(f"Object Type: {last_object_type}")
print(f"Full Test Input Sequence: {last_sequence}")
print(f"Actual Next Activity (Ground Truth): {actual_next_activity}")
print(f"Predicted Next Activity (Decision Tree): {predicted_next_activity_dt} {'✅' if correct_dt else '❌'}")
print(f"Predicted Next Activity (Random Forest): {predicted_next_activity_rf} {'✅' if correct_rf else '❌'}")
print(f"Predicted Next Activity (Neural Network): {predicted_next_activity_nn} {'✅' if correct_nn else '❌'}")

# === EVALUATE MODEL PERFORMANCE ===
y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_nn = nn_model.predict(X_test)

print("\n=== MODEL PERFORMANCE ===")
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_nn))

print("\n=== CLASSIFICATION REPORT ===")
print("Decision Tree Report:\n", classification_report(y_test, y_pred_dt, target_names=label_encoder.classes_, zero_division=0))
print("Random Forest Report:\n", classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_, zero_division=0))
print("Neural Network Report:\n", classification_report(y_test, y_pred_nn, target_names=label_encoder.classes_, zero_division=0))