import argparse
import os
import torch
import cv2
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# ==========================================
# 1. DEFINE CLASSES
# ==========================================

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 1. Check if channels (39) are at the end. If so, move them to dim 1.
        if x.ndim == 4 and x.shape[3] == 39:
            x = x.permute(0, 3, 1, 2)  
        # 2. Check if input is flat [Batch, 39]. If so, reshape.
        elif x.ndim == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        # 3. Expand to VQGAN Latent Size (32x32)
        target_h, target_w = 32, 32
        if x.shape[-1] != target_w or x.shape[-2] != target_h:
             x = x.expand(-1, -1, target_h, target_w)
        return x

class SLPPoseBase(Dataset):
    def __init__(self, txt_file, data_root, target_dir="target", label_dir="labels", size=None, interpolation="bicubic", flip_p=0.0):
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
            filtered_triplets = [all_triplets[0]] + all_triplets[5:]
        else:
            filtered_triplets = all_triplets

        final_values = []
        for triplet in filtered_triplets:
            final_values.extend(triplet)

        count = min(len(final_values), 39)
        flat_kpts[:count] = final_values[:count]
        return flat_kpts

    def __getitem__(self, i):
        rel_path = self.image_paths[i]
        
        target_path = os.path.join(self.data_root, self.target_dir, rel_path)
        txt_filename = os.path.splitext(rel_path)[0] + ".txt"
        label_path   = os.path.join(self.data_root, self.label_dir, txt_filename)

        target_img = Image.open(target_path).convert("RGB")
        if self.size is not None:
            target_img = target_img.resize((self.size, self.size), resample=self.interpolation)
        
        pose_vector = self.get_raw_keypoints(label_path)
        pose_vector = torch.from_numpy(pose_vector).float()
        pose_vector = pose_vector.reshape(1, 1, 39)

        target_img = np.array(target_img).astype(np.uint8)
        target_norm = (target_img / 127.5 - 1.0).astype(np.float32)

        return {
            "image": target_norm, 
            "pose": pose_vector, 
            "txt": "",
            "fname": txt_filename # <--- PASS FILENAME TO DATALOADER
        }

class SLPPoseValidation(SLPPoseBase):
    def __init__(self, txt_file="data/slp-conditional/val.txt", data_root="data/slp-conditional", target_dir="target", label_dir="labels", **kwargs):
        super().__init__(txt_file=txt_file,
                         data_root=data_root,
                         target_dir=target_dir,
                         label_dir=label_dir,
                         **kwargs)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}...")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def save_pose_visualization(pose_tensor, out_path, height, width):
    data = pose_tensor.cpu().numpy().flatten()
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    points = []
    for i in range(0, len(data), 3):
        x_norm = data[i]
        y_norm = data[i+1]
        vis    = data[i+2]
        x_px = int(x_norm * width)
        y_px = int(y_norm * height)
        points.append((x_px, y_px, vis))

    limbs = [
        (1, 2), (1, 7), (2, 8), (7, 8),   
        (1, 3), (3, 5),                   
        (2, 4), (4, 6),                   
        (7, 9), (9, 11),                  
        (8, 10), (10, 12)                 
    ]
    color_limb = (0, 255, 0) 
    color_joint = (0, 0, 255) 

    for p1, p2 in limbs:
        if p1 < len(points) and p2 < len(points):
            pt_a = points[p1]
            pt_b = points[p2]
            if pt_a[2] > 0 and pt_b[2] > 0:
                cv2.line(canvas, (pt_a[0], pt_a[1]), (pt_b[0], pt_b[1]), color_limb, 2)

    for x, y, v in points:
        if v > 0:
            cv2.circle(canvas, (x, y), 3, color_joint, -1)

    Image.fromarray(canvas).save(out_path)

# ==========================================
# 3. RUN INFERENCE
# ==========================================

def get_parser():
    parser = argparse.ArgumentParser(description="SLP full keypoints no-MLP concat sampler")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--val-txt", default="data/slp-conditional/val.txt", help="Validation txt file path")
    parser.add_argument("--data-root", default="data/slp-conditional", help="Dataset root path")
    parser.add_argument("--target-dir", default="target", help="Target image subdirectory under data root")
    parser.add_argument("--label-dir", default="labels", help="Label subdirectory under data root")
    parser.add_argument("--num-variations", type=int, default=1, help="Samples per conditioning input")
    parser.add_argument("--ddim-steps", type=int, default=100, help="DDIM steps")
    parser.add_argument("--scale", type=float, default=1.0, help="Classifier-free guidance scale")
    parser.add_argument("--ddim-eta", type=float, default=1.0, help="DDIM eta")
    parser.add_argument("--image-size", type=int, default=256, help="Input image resize")
    parser.add_argument("--num-workers", type=int, default=1, help="Dataloader workers")
    return parser


def run_inference(args):
    config_path = args.config
    ckpt_path = args.ckpt
    outdir = args.outdir
    num_variations = args.num_variations
    ddim_steps = args.ddim_steps
    scale = args.scale
    ddim_eta = args.ddim_eta
    image_size = args.image_size

    os.makedirs(outdir, exist_ok=True)
    config = OmegaConf.load(config_path)

    print(f"Overriding conditioner target to use local Identity class...")
    config.model.params.cond_stage_config.target = "__main__.Identity"

    model = load_model_from_config(config, ckpt_path)
    sampler = DDIMSampler(model)

    dataset = SLPPoseValidation(txt_file=args.val_txt,
                                data_root=args.data_root,
                                target_dir=args.target_dir,
                                label_dir=args.label_dir,
                                size=image_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    print(f"Generating {num_variations} variations per image...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Sampling")):
            
            batch['pose'] = batch['pose'].cuda()
            
            # --- EXTRACT FILENAME ---
            # batch['fname'] is a tuple of filenames because batch_size=1
            original_fname = batch['fname'][0]
            # Remove .txt extension to get the name stem
            name_stem = os.path.splitext(original_fname)[0]
            # ------------------------

            # # Save Skeleton
            # skel_filename = f"{name_stem}_skeleton.png"
            # skel_path = os.path.join(outdir, skel_filename)
            # save_pose_visualization(batch['pose'], skel_path, image_size, image_size)
            
            # Get Conditioning
            c = model.get_learned_conditioning(batch['pose'])
            
            # Loop for diversity
            for n in range(num_variations):
                shape = (model.channels, model.image_size, model.image_size)
                x_T = torch.randn((1, *shape), device=model.device)

                samples, _ = sampler.sample(S=ddim_steps,
                                            conditioning=c,
                                            batch_size=1,
                                            shape=shape,
                                            x_T=x_T,
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            eta=ddim_eta)

                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                
                img_tensor = x_samples[0]
                img_tensor = img_tensor.permute(1, 2, 0).cpu().numpy()
                img_array = (img_tensor * 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                
                # Naming Logic:
                # If doing 1 variation, use the exact label name: "image_01.png"
                # If doing multiple, append variation index: "image_01_v0.png"
                if num_variations > 1:
                    filename = f"{name_stem}_v{n}.png"
                else:
                    filename = f"{name_stem}.png"
                    
                img.save(os.path.join(outdir, filename))

if __name__ == "__main__":
    run_inference(get_parser().parse_args())