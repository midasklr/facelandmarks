import torch
import os
from faceland import FaceLanndInference

checkpoint = torch.load('faceland.pth')
model = PFLDInference()
model.load_state_dict(checkpoint)
model.eval()

print(model)
example = torch.rand(1, 3, 112, 112)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("ldm_s.pt")
