import torch
import torch.nn as nn
import torch.optim as optim

x = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(5):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

test_input = torch.randn(1, 10)
output = model(test_input)
prediction = torch.argmax(output, dim=1)
print("Test input:", test_input)
print("Output:", output)
print("Prediction:", prediction)
