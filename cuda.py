import torch
print("torch._version:", torch.version_)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device(), torch.cuda.get_device_name(0))