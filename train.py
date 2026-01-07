import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split


# Load CSV and convert to tensors (repeat from your first script)
df = pd.read_csv('synthetic_data.csv')
X = df[['Feature1', 'Feature2']].values
y = df['Label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define logistic regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # output is 1 for binary classification

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # sigmoid for probability output

# Initialize model, loss, optimizer
model = LogisticRegressionModel(input_dim=2)
criterion = nn.BCELoss()  # Binary Cross Entropy loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


#model.train() sets the model to training mode 
#optimizer.zero_grad() clears gradients from the previous step to avoid accumulation.
#outputs = model(X_train_tensor) runs a forward pass to get predicted probabilities for all training samples.
#loss = criterion(outputs, y_train_tensor) calculates how far predictions are from true labels.
#loss.backward() computes gradients of the loss with respect to model parameters using backpropagation.
#optimizer.step() updates the model parameters using the gradients and learning rate.
#Every 10 epochs, it prints the current loss to monitor training progress.




# Evaluate on test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    predicted = (y_pred >= 0.5).float()
    accuracy = (predicted == y_test_tensor).float().mean()
    print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
