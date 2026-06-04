from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn

print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())

print()

# Load and prepare data
# =====================

iris = load_iris()
X = iris.data
y = iris.target

print("X sample:")
print(X[:3])

print()

print("y sample:")
print(y[:3])

print()

# Hold back data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

# Scale features (mean=0, sd=1)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define the model
# ================

#   4   ->   8   ->   12  ->  8  ->  3
# input       3x hidden layers     output
#             (ReLU activation)

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),  # activation function
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 8),
    nn.ReLU(),
    nn.Linear(8, 3),
)

print("Created model:")
print(model)

print()

print("Layer 1 weights shape:", model[0].weight.shape)
print("Layer 1 biases shape:", model[0].bias.shape)

print()

print("Randomly initialised weights:")
print(model[0].weight.data)

print()

print("Total parameters:", sum(p.numel() for p in model.parameters()))

print()

print("3x example predictions of untrained model:")

print("Inputs:")
inputs = X_train[:3]
print(inputs)

print()

print("Raw outputs:")
outputs = model(inputs)
print(outputs)

print()

print("Scaled outputs:")
print(torch.softmax(outputs, dim=1))

print()

print("Predicted classes:")
predictions = outputs.argmax(dim=1)
print(predictions)
print(iris.target_names[predictions])

print()

# Loss function and optimiser
# ===========================

loss_fn = nn.CrossEntropyLoss()  # for multi-class classification
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam with learning rate = 0.01

# Training loop
# =============

n_epochs = 100  # train on entire dataset 100 times
losses = []     # keep track of loss each epoch for later plotting

print("Starting training...")
for epoch in range(n_epochs):
    # Forward pass
    output = model(X_train)

    # Measure how wrong we are
    loss = loss_fn(output, y_train)
    losses.append(loss.item())  # record for later plotting

    # Backpropagation
    optimiser.zero_grad()  # clear gradients (from last loop)
    loss.backward()        # calculate gradients by backpropagation
    optimiser.step()       # update weights

    # Report progress
    print("Epoch", epoch, "loss =", loss.item())

print("Training complete")

print()

# Evaluation
# ==========

with torch.no_grad():
    output = model(X_test)
    y_pred = output.argmax(dim=1)  # predict the class with largest value
    accuracy = (y_pred == y_test).float().mean() * 100

print("Test accuracy:", accuracy.item(), "%")

# Diagnostic plots
# ================

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Training loss over time
fig, ax = plt.subplots()
ax.plot(losses)
ax.set(title="Training loss over time", xlabel="Epoch", ylabel="Loss")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot()

plt.show()
