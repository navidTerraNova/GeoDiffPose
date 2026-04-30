import os
import numpy as np
import torch
import torch.nn as nn
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# --- THE EMBEDDER ---
class PoseEmbedder(nn.Module):
    def __init__(self, input_dim=39, embed_dim=128):
        super().__init__()
        # A simple MLP to "learn" the pose representation
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        # x input shape: (Batch, 39) or (Batch, 39, 1, 1) due to LDM loading
        
        # 1. Flatten to (Batch, 39)
        x = x.view(x.size(0), -1) 
        
        # 2. Project to Embedding (Batch, 128)
        out = self.net(x.float())
        
        # 3. SPATIAL BROADCAST
        # We need to reshape this vector into a feature map of size (32, 32)
        # to match the VQGAN latent spatial dimensions.
        
        # Reshape to (Batch, 128, 1, 1)
        out = out.unsqueeze(-1).unsqueeze(-1)
        
        # Expand to (Batch, 128, 32, 32)
        out = out.expand(-1, -1, 32, 32)
        
        return out
# --------------------

class SLPPoseBase(Dataset):
    def __init__(self, txt_file, data_root, target_dir="target", label_dir="labels", 
                 size=None, interpolation="bicubic", flip_p=0.5):
        self.data_paths = txt_file
        self.data_root = data_root
        self.target_dir = target_dir
        self.label_dir = label_dir
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.size = size
        self.interpolation = {
            "linear": PIL.Image.Resampling.BILINEAR,
            "bilinear": PIL.Image.Resampling.BILINEAR,
            "bicubic": PIL.Image.Resampling.BICUBIC,
            "lanczos": PIL.Image.Resampling.LANCZOS,
        }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def get_raw_keypoints(self, txt_path):
        # ... (Same logic as before: Parse 13 keypoints * 3 = 39 values) ...
        flat_kpts = np.zeros(39, dtype=np.float32)
        if not os.path.exists(txt_path): return flat_kpts
        with open(txt_path, "r") as f: content = f.read().strip().split()
        if not content: return flat_kpts
        
        raw_values = content[5:]
        all_triplets = []
        for i in range(0, len(raw_values), 3):
            try:
                x = float(raw_values[i])
                y = float(raw_values[i+1])
                v = float(raw_values[i+2]) / 2.0
                all_triplets.append([x, y, v])
            except IndexError: break
        
        if len(all_triplets) >= 5:
            filtered = [all_triplets[0]] + all_triplets[5:]
        else:
            filtered = all_triplets

        final_values = []
        for t in filtered: final_values.extend(t)
        count = min(len(final_values), 39)
        flat_kpts[:count] = final_values[:count]
        return flat_kpts

    def __getitem__(self, i):
        rel_path = self.image_paths[i]
        target_path = os.path.join(self.data_root, self.target_dir, rel_path)
        txt_filename = os.path.splitext(rel_path)[0] + ".txt"
        label_path   = os.path.join(self.data_root, self.label_dir, txt_filename)

        # Load Image
        target_img = Image.open(target_path).convert("RGB")
        if self.size is not None:
            target_img = target_img.resize((self.size, self.size), resample=self.interpolation)
        
        # Load Raw Vector
        pose_vector = self.get_raw_keypoints(label_path)
        pose_vector = torch.from_numpy(pose_vector).float()
        
        # Reshape to "Fake Image" so DataLoader doesn't complain about dims
        pose_vector = pose_vector.reshape(1, 1, 39)

        target_img = np.array(target_img).astype(np.uint8)
        target_norm = (target_img / 127.5 - 1.0).astype(np.float32)

        return {
            "image": target_norm, 
            "pose": pose_vector, 
            "txt": ""
        }

class SLPPoseTrain(SLPPoseBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/slp-conditional/train.txt", data_root="data/slp-conditional", **kwargs)

class SLPPoseValidation(SLPPoseBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file="data/slp-conditional/val.txt", data_root="data/slp-conditional", flip_p=flip_p, **kwargs)