import torch
import torch.nn as nn
from SmartAITool.core import *



x = torch.randn(1, 1, 8)
y = torch.randn(1, 16, 4)


identity = nn.Identity()
cprint(identity)
print(identity(x).shape)
print(identity)
bprint()
print(x)


z = x + y 
cprint(z)

