import os
import numpy as np
import torch
import torch.nn as nn
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# --- IDENTITY PASSTHROUGH ---
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Input x comes in as (Batch, 26, 1, 1) thanks to the dataset reshape + LDM rearrange.
        
        # Ensure we expand to the VQGAN latent size (32x32)
        target_h, target_w = 32, 32
        
        # Check if expansion is needed
        if x.shape[-1] != target_w or x.shape[-2] != target_h:
             # Expand: (B, 26, 1, 1) -> (B, 26, 32, 32)
             x = x.expand(-1, -1, target_h, target_w)
             
        return x

class SLPPoseBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 target_dir="target",
                 label_dir="labels",
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
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
        """
        Parses format: [Class, BBox... Kpt1_x, Kpt1_y, Kpt1_v, ... ]
        
        Filters out redundant head points (indices 1, 2, 3, 4).
        Returns XY ONLY: 13 keypoints * 2 values = 26 floats.
        """
        # 13 keypoints * 2 values (XY) = 26 floats
        flat_kpts = np.zeros(26, dtype=np.float32)

        if not os.path.exists(txt_path):
            return flat_kpts

        with open(txt_path, "r") as f:
            content = f.read().strip().split()
            
        if not content:
            return flat_kpts

        # 1. Skip Metadata (Class + BBox)
        raw_values = content[5:]
        
        # 2. Group into pairs [x, y], IGNORING v
        all_pairs = []
        # We still step by 3 because the text file likely has (x, y, v) structure
        for i in range(0, len(raw_values), 3):
            try:
                x = float(raw_values[i])
                y = float(raw_values[i+1])
                # We skip v at i+2
                all_pairs.append([x, y])
            except IndexError:
                break
        
        # 3. Filter Redundant Points (Same logic as before)
        # Keep index 0, drop 1,2,3,4. Keep 5+
        if len(all_pairs) >= 5:
            filtered_pairs = [all_pairs[0]] + all_pairs[5:]
        else:
            filtered_pairs = all_pairs

        # 4. Flatten back to a single list
        final_values = []
        for pair in filtered_pairs:
            final_values.extend(pair)

        # 5. Fill numpy array
        count = min(len(final_values), 26)
        flat_kpts[:count] = final_values[:count]
                
        return flat_kpts

    def __getitem__(self, i):
            rel_path = self.image_paths[i]
            
            # Paths configured in config file
            target_path = os.path.join(self.data_root, self.target_dir, rel_path)
            txt_filename = os.path.splitext(rel_path)[0] + ".txt"
            label_path   = os.path.join(self.data_root, self.label_dir, txt_filename)

            # 1. Load Image
            target_img = Image.open(target_path).convert("RGB")
            if self.size is not None:
                target_img = target_img.resize((self.size, self.size), resample=self.interpolation)
            
            # 2. Load Coordinates (26 values)
            pose_vector = self.get_raw_keypoints(label_path)
            pose_vector = torch.from_numpy(pose_vector).float()
            
            # --- RESHAPE TO FAKE IMAGE (1, 1, 26) ---
            # Dimensions: Height=1, Width=1, Channels=26
            pose_vector = pose_vector.reshape(1, 1, 26) 
            # ----------------------------------------

            # Normalize Image
            target_img = np.array(target_img).astype(np.uint8)
            target_norm = (target_img / 127.5 - 1.0).astype(np.float32)

            return {
                "image": target_norm, 
                "pose": pose_vector, 
                "txt": ""
            }

class SLPPoseTrain(SLPPoseBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/slp-conditional/train.txt", 
                         data_root="data/slp-conditional", 
                         **kwargs)

class SLPPoseValidation(SLPPoseBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file="data/slp-conditional/val.txt", 
                         data_root="data/slp-conditional",
                         flip_p=flip_p, 
                         **kwargs)