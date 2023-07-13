import torch

print("Is MacOS version ≥ 12.3?\t", torch.backends.mps.is_available())
device = torch.device("mps")
print("Device available:\t\t\t", device)

x = torch.rand(3, 3)
print("x:\t\t\t\t", x)
x.to(device)
print("Successfully moved x to", device)