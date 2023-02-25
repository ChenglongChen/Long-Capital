import torch
import os
import glob
all_files = sorted(glob.glob("./checkpoints/*"))
for path in all_files:
    if "latest" in path:
        continue
    real_path = os.path.realpath(path)
    print(torch.load(real_path)['metrics'])
