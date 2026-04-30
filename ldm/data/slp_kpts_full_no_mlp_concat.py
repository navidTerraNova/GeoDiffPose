import os
import numpy as np
import torch
import torch.nn as nn
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# --- IDENTITY PASSTHROUGH (Fixed for broadcasting) ---
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Input x comes in as (Batch, 39, 1, 1) thanks to the dataset reshape + LDM rearrange.
        
        # Ensure we expand to the VQGAN latent size (32x32)
        target_h, target_w = 32, 32
        
        # Check if expansion is needed
        if x.shape[-1] != target_w or x.shape[-2] != target_h:
             # Expand: (B, 39, 1, 1) -> (B, 39, 32, 32)
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
        Parses the specific format:
        [Class, BBox1, BBox2, BBox3, BBox4, Kpt1_x, Kpt1_y, Kpt1_v, Kpt2... ]
        
        Filters out redundant head points (indices 1, 2, 3, 4 of keypoints).
        Returns a flat numpy array of shape (39,).
        """
        # 13 keypoints * 3 values = 39 floats
        flat_kpts = np.zeros(39, dtype=np.float32)

        if not os.path.exists(txt_path):
            return flat_kpts

        with open(txt_path, "r") as f:
            content = f.read().strip().split()
            
        if not content:
            return flat_kpts

        # 1. Skip Metadata
        # Index 0 is Class, Index 1-4 is BBox. 
        # Keypoints start at index 5.
        raw_values = content[5:]
        
        # 2. Group into triplets [x, y, v]
        all_triplets = []
        for i in range(0, len(raw_values), 3):
            try:
                x = float(raw_values[i])
                y = float(raw_values[i+1])
                v = float(raw_values[i+2]) / 2.0
                all_triplets.append([x, y, v])
            except IndexError:
                break
        
        # 3. Filter Redundant Points
        # We assume the first 5 triplets (indices 0,1,2,3,4) are the head cluster.
        # We keep index 0, and drop 1,2,3,4. Then keep the rest (5+).
        if len(all_triplets) >= 5:
            # Slice: [First item] + [Item 5 to End]
            filtered_triplets = [all_triplets[0]] + all_triplets[5:]
        else:
            # Fallback if data is short
            filtered_triplets = all_triplets

        # 4. Flatten back to a single list
        final_values = []
        for triplet in filtered_triplets:
            final_values.extend(triplet)

        # 5. Fill numpy array safely
        count = min(len(final_values), 39)
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
            
            # 2. Load Coordinates (39 values)
            pose_vector = self.get_raw_keypoints(label_path)
            pose_vector = torch.from_numpy(pose_vector).float()
            
            # --- FIX: RESHAPE TO FAKE IMAGE (1, 1, 39) ---
            # This gives it 3 dimensions: Height=1, Width=1, Channels=39.
            # The DataLoader adds Batch, making it (Batch, 1, 1, 39).
            # The LDM code will rearrange this to (Batch, 39, 1, 1), which is what we want.
            pose_vector = pose_vector.reshape(1, 1, 39) 
            # ---------------------------------------------

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