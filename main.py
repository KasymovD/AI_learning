import os
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

torch.manual_seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"

n_samples = 2000
n_features = 20
n_classes = 2

X = torch.randn(n_samples, n_features)
true_w = torch.randn(n_features, 1)
logits = X @ true_w + 0.25 * torch.randn(n_samples, 1)
probs = torch.sigmoid(logits).squeeze(1)
Y = (probs > 0.5).long()

idx = torch.randperm(n_samples)
train_end = math.floor(0.7 * n_samples)
val_end = math.floor(0.85 * n_samples)
train_idx, val_idx, test_idx = idx[:train_end], idx[train_end:val_end], idx[val_end:]

X_train, y_train = X[train_idx], Y[train_idx]
X_val, y_val = X[val_idx], Y[val_idx]
X_test, y_test = X[test_idx], Y[test_idx]

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return self.x.size(0)
    def __getitem__(self, i):
        return self.x[i], self.y[i]

batch_size = 64
train_loader = torch.utils.data.DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
val_loader = torch.utils.data.DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False, drop_last=False)
test_loader = torch.utils.data.DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=2, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

model = MLP(n_features, hidden=64, out_dim=n_classes, p=0.2).to(device)
if hasattr(torch, "compile"):
    try:
        model = torch.compile(model, dynamic=False)
    except Exception:
        pass

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=False)

scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

epochs = 50
grad_clip = 1.0
patience = 7
best_val = float("inf")
best_path = "best_model.pt"
wait = 0

def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()
    losses = []
    all_y = []
    all_p = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(xb)
                loss = criterion(out, yb)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
        losses.append(loss.detach().item())
        probs = torch.softmax(out.detach(), dim=1)[:, 1]
        all_p.append(probs.cpu())
        all_y.append(yb.detach().cpu())
    y_true = torch.cat(all_y).numpy()
    y_prob = torch.cat(all_p).numpy()
    y_pred = (y_prob >= 0.5).astype("int64")
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    return float(sum(losses) / len(losses)), acc, auc

for epoch in range(1, epochs + 1):
    train_loss, train_acc, train_auc = run_epoch(train_loader, train=True)
    val_loss, val_acc, val_auc = run_epoch(val_loader, train=False)
    scheduler.step(val_loss)
    lr_now = optimizer.param_groups[0]["lr"]
    print(f"epoch {epoch:02d} | lr {lr_now:.5f} | train_loss {train_loss:.4f} acc {train_acc:.4f} auc {train_auc:.4f} | val_loss {val_loss:.4f} acc {val_acc:.4f} auc {val_auc:.4f}")
    if val_loss < best_val - 1e-6:
        best_val = val_loss
        wait = 0
        torch.save(model.state_dict(), best_path)
    else:
        wait += 1
        if wait >= patience:
            print("early stopping")
            break

if os.path.exists(best_path):
    state_dict = torch.load(best_path, map_location=device, weights_only=False) if "weights_only" in torch.load.__code__.co_varnames else torch.load(best_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except Exception:
        if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
            base = model._orig_mod if hasattr(model, "_orig_mod") else model
            base.load_state_dict(state_dict)

model.eval()
test_loss, test_acc, test_auc = run_epoch(test_loader, train=False)
print(f"test_loss {test_loss:.4f} | test_acc {test_acc:.4f} | test_auc {test_auc:.4f}")

ys = []
ps = []
for xb, yb in test_loader:
    with torch.no_grad():
        out = model(xb.to(device))
        prob1 = torch.softmax(out, dim=1)[:, 1].cpu()
    ys.append(yb.cpu())
    ps.append(prob1)
y_true = torch.cat(ys).numpy()
y_prob = torch.cat(ps).numpy()
y_pred = (y_prob >= 0.5).astype("int64")

print("confusion_matrix")
print(confusion_matrix(y_true, y_pred))
print("classification_report")
print(classification_report(y_true, y_pred, digits=4))

with torch.no_grad():
    sample = torch.randn(1, n_features).to(device)
    logits = model(sample)
    pred = torch.argmax(logits, dim=1).item()
    prob = torch.softmax(logits, dim=1)[0, pred].item()
print("sample_prediction", {"class": int(pred), "confidence": float(prob)})
