import torch

a = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]], requires_grad=True)

b = torch.tensor([[7., 8.],
                  [9., 10.],
                  [11., 12.]], requires_grad=True)

c = a @ b
d = torch.sum(c)

print("a:", a)
print("b:", b)
print("c:", c)
print("d:", d)

d.backward()

print("Grad a:", a.grad)
print("Grad b:", b.grad)

x = torch.randn(3, 3)
print("x:", x)
print("x mean:", x.mean())
print("x std:", x.std())

x_reshaped = x.view(1, 9)
print("x reshaped:", x_reshaped)

y = torch.linspace(0, 1, steps=5)
print("y:", y)

z = torch.rand((2, 2))
print("z:", z)

device = "cuda" if torch.cuda.is_available() else "cpu"
t = torch.ones((3, 3), device=device)
print("t:", t)
