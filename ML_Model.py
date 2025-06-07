import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier

# Load the crop recommendation dataset as a Pandas DataFrame
org_data = pd.read_csv("Crop_Recommendation.csv")

# Convert the DataFrame into a NumPy array for faster processing
data = np.array(org_data)
m, n = data.shape       # (2200, 8)

# Data reproducibility
random.seed(42)
random.shuffle(data)

# Test Data --- 20% of total datapoints
data_test = data[0:m//5]         # shape: (440, 8)
X_test = data_test[:, 0:n-1]     # shape: (440, 7)
Y_test = data_test[:, -1]        # shape: (440,)

# Training Data --- 80% of total datapoints
data_train = data[m//5:]         # shape: (1760, 8)
X_train = data_train[:, 0:n-1]   # shape: (1760, 7)
Y_train = data_train[:, -1]      # shape: (1760,)

# Initializing the model
model = RandomForestClassifier()

# Train the model with the training data
model.fit(X_train, Y_train)

# ---------- 🔥 WHAT IS PICKLE? 🔥 ----------

# 'pickle' is a built-in Python module used to save Python objects to a file
# It "serializes" the model — which means converting it into a byte stream
# This lets you save the trained model to a file, and load it later without retraining

import pickle

# Save the trained model into a file named 'ML_Model.pkl'
# 'wb' stands for 'write binary' — we are writing binary data to the file
pickle.dump(model, open("ML_Model.pkl", "wb"))

# Now, 'ML_Model.pkl' contains your trained model and can be loaded later like this:
# model = pickle.load(open("ML_Model.pkl", "rb"))
