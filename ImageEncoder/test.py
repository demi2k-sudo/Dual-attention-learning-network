import torch

# Check if cpu is available
cpu_available = torch.cpu.is_available()

print(cpu_available)
