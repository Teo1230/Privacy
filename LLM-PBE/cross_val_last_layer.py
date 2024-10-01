import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Step 1: Load the JSON files
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Load good and bad data
good_data = load_json('good_emb.json')
print(f"Number of good entries: {len(good_data)}")
bad_data = load_json('bad_emb.json')
print(f"Number of bad entries: {len(bad_data)}")

# Step 2: Prepare the data
good_embs = [entry['emb'] for entry in good_data]
bad_embs = [entry['emb'] for entry in bad_data]

# Create labels: 1 for good, 0 for bad
X = np.array(good_embs + bad_embs)
y = np.array([1] * len(good_embs) + [0] * len(bad_embs))

# Step 3: Perform Logistic Regression with 5-fold cross-validation
model = make_pipeline(StandardScaler(), LogisticRegression())

# Perform cross-validation
scores = cross_val_score(model, X.tolist(), y, cv=10)
# Display the results
print("Cross-validation scores for each fold:", scores)
print("Mean cross-validation score:", np.mean(scores))
