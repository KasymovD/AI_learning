import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(100, 10).to(device)
y = torch.randint(0, 2, (100,)).to(device)

class SimpleNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=2):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

model = SimpleNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    preds = torch.argmax(y_pred, dim=1)
    acc = accuracy_score(y.cpu(), preds.cpu())
    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}")

model.eval()
with torch.no_grad():
    test_input = torch.randn(1, 10).to(device)
    output = model(test_input)
    prediction = torch.argmax(output, dim=1)

print("\nTest Input:", test_input)
print("Raw Output (logits):", output)
print("Predicted Class:", prediction.item())
