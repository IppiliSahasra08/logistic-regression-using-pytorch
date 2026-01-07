import torch
import numpy as np

# Create dummy numpy arrays
X_train = np.random.rand(140, 2)
y_train = np.random.randint(0, 2, size=(140,))

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Print shapes
print("X_train_tensor shape:", X_train_tensor.shape, flush=True)
print("y_train_tensor shape:", y_train_tensor.shape, flush=True)
