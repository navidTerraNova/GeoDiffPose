import os
import numpy as np
import torch
import torch.nn as nn
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# --- THE CROSS-ATTENTION EMBEDDER ---
class PoseEmbedder(nn.Module):
    # CHANGED: Default input_dim from 39 to 26 (13 keypoints * 2 coordinates)
    def __init__(self, input_dim=26, embed_dim=512):
        super().__init__()
        # MLP to project raw coordinates into the transformer context dimension
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        # x input shape: (Batch, 26) or (Batch, 1, 1, 26)
        
        # 1. Flatten to (Batch, 26)
        # We ensure it's a flat vector before projecting
        x = x.view(x.size(0), -1) 
        
        # 2. Project to Embedding (Batch, 512)
        out = self.net(x.float())
        
        # 3. SEQUENCE FORMATTING
        # Cross-Attention expects a sequence of tokens: [Batch, Sequence_Len, Dim]
        # We treat the entire pose as a single token (Seq_Len = 1).
        return out.unsqueeze(1) 
# ------------------------------------

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
        # CHANGED: Logic for 26 keypoints (13 points * 2 coordinates: x, y)
        flat_kpts = np.zeros(26, dtype=np.float32)
        if not os.path.exists(txt_path): return flat_kpts
        with open(txt_path, "r") as f: content = f.read().strip().split()
        if not content: return flat_kpts
        
        raw_values = content[5:]
        all_pairs = []
        
        # We still iterate by 3 if the source file format is x,y,v
        # But we only keep x and y
        for i in range(0, len(raw_values), 3):
            try:
                x = float(raw_values[i])
                y = float(raw_values[i+1])
                # We skip 'v' (visibility) for XY mode
                all_pairs.append([x, y])
            except IndexError: break
        
        # Filter redundant points (Keep index 0, drop 1-4, keep 5+)
        if len(all_pairs) >= 5:
            filtered = [all_pairs[0]] + all_pairs[5:]
        else:
            filtered = all_pairs

        final_values = []
        for p in filtered: final_values.extend(p)
        
        # CHANGED: Limit count to 26
        count = min(len(final_values), 26)
        flat_kpts[:count] = final_values[:count]
        return flat_kpts

    def __getitem__(self, i):
        rel_path = self.image_paths[i]
        
        # Paths configured in config file
        target_path = os.path.join(self.data_root, self.target_dir, rel_path)
        txt_filename = os.path.splitext(rel_path)[0] + ".txt"
        label_path   = os.path.join(self.data_root, self.label_dir, txt_filename)

        # Load Image
        target_img = Image.open(target_path).convert("RGB")
        if self.size is not None:
            target_img = target_img.resize((self.size, self.size), resample=self.interpolation)
        
        # Load Vector
        pose_vector = self.get_raw_keypoints(label_path)
        pose_vector = torch.from_numpy(pose_vector).float()
        
        # CHANGED: Reshape to "Fake Image" (1, 1, 26)
        pose_vector = pose_vector.reshape(1, 1, 26)

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