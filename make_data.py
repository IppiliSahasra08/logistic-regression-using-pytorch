import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

# Combine features and labels into one DataFrame
df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
df['Label'] = y

# Save to CSV
df.to_csv('synthetic_data.csv', index=False)
print("Dataset saved to synthetic_data.csv")

# Load the dataset back
loaded_df = pd.read_csv('synthetic_data.csv')
X_loaded = loaded_df[['Feature1', 'Feature2']].values
y_loaded = loaded_df['Label'].values

print("Loaded X shape:", X_loaded.shape)
print("Loaded y shape:", y_loaded.shape)

# Split into train and test sets (e.g., 70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_loaded, y_loaded, test_size=0.3, random_state=42, stratify=y_loaded
)

print("Train set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Visualize train and test sets
plt.figure(figsize=(10, 5))

# Plot training data
plt.subplot(1, 2, 1)
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='blue', label='Class 0')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='Class 1')
plt.title('Training Set')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

# Plot testing data
plt.subplot(1, 2, 2)
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='blue', label='Class 0')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='red', label='Class 1')
plt.title('Testing Set')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
