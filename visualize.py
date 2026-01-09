import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from train import LogisticRegressionModel  # or redefine model here

# Load dataset
df = pd.read_csv("synthetic_data.csv")
X = df[['Feature1', 'Feature2']].values
y = df['Label'].values

# Load trained model
model = LogisticRegressionModel(input_dim=2)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Create grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(grid, dtype=torch.float32)

# Predict
with torch.no_grad():
    probs = model(grid_tensor).reshape(xx.shape)

# Plot
plt.contourf(xx, yy, probs, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("Decision Boundary (Logistic Regression)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
