import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SLPBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
            
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        self.size = size
        
        # Pillow 10+ compatible interpolation
        self.interpolation = {
            "linear": PIL.Image.Resampling.BILINEAR,
            "bilinear": PIL.Image.Resampling.BILINEAR,
            "bicubic": PIL.Image.Resampling.BICUBIC,
            "lanczos": PIL.Image.Resampling.LANCZOS,
        }[interpolation]
        
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        
        # 1. Open Image
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # 2. Resize Whole Image (No Cropping)
        # If size is set (e.g. 256), this squishes 1024x576 -> 256x256
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        # 3. Augmentations
        image = self.flip(image)
        
        # 4. Normalize
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        
        return example


class SLPTrain(SLPBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/slp/train.txt", 
                         data_root="data/slp", 
                         **kwargs)


class SLPValidation(SLPBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file="data/slp/val.txt", 
                         data_root="data/slp",
                         flip_p=flip_p, 
                         **kwargs)