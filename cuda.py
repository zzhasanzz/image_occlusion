import torch

# FIX: Changed torch.version_ to torch.__version__
print("torch version:", torch.__version__) 
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    # Minor fix: Added a comma for better formatting
    print("current device:", torch.cuda.current_device(), ",", torch.cuda.get_device_name(0))