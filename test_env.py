import torch
import torch_geometric


print('torch.__version__:', torch.__version__)
print('torch.cuda.is_available():', torch.cuda.is_available())
print('torch.cuda.get_device_name(0):', torch.cuda.get_device_name(0))
print('torch.cuda.device_count():', torch.cuda.device_count())
print('torch.version.cuda:', torch.version.cuda)

print('torch_geometric.__version__:', torch_geometric.__version__)
