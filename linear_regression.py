import torch
import torch.nn as nn

# 1. Define input size, output_size, forward pass
# 2. define loss function and optimizer
# 3. Train the neural network
    # 3.1 forward pass
    # 3.2 calculate loss
    # 3.3 update weights

# f = w * x + b

X = torch.tensor([[1],[2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape

# model prediction
model = nn.Linear(n_features, n_features)


print(f"Prediction before training: f(5) = {model(test).item():.3f}")

LR = 0.01
epochs = 100
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
for epoch in range(epochs):
    # train
    # 3.1
    y_pred = model(X)
    # 3.2
    l = loss(Y, y_pred)
    l.backward() #calculates the gradient on its own

    # update weights using stochastic gradient descent
    # 3.3
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 2 == 1:
        [w, b] = model.parameters()
        print(f"epoch{epoch + 1}: w = {w[0][0].item()}, loss = {l:.3f}")
print(f"Prediction after training: f(5) = {model(test).item():.3f}")
