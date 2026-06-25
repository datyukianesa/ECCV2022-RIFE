import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='RIFE Interpolation for ATD-12K Dataset')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
args = parser.parse_args()

try:
    try:
        try:
            from model.RIFE_HDv2 import Model
            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v2.x HD model.")
        except:
            from train_log.RIFE_HDv3 import Model
            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v3.x HD model.")
    except:
        from model.RIFE_HD import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v1.x HD model")
except:
    from model.RIFE import Model
    model = Model()
    model.load_model(args.modelDir, -1)
    print("Loaded ArXiv-RIFE model")

model.eval()
model.device()

#DATASET PATH
dataset_dir = Path("/content/datasets/test_2k_540p")
print(f"Starting inference on dataset: {dataset_dir}")

count = 0

#Inference loop
for folder in dataset_dir.iterdir():
    if folder.is_dir():
        img0_path = folder / "frame1.png"
        img1_path = folder / "frame3.png"
        
        save_path = folder / "frame2_rife.png"

        if img0_path.exists() and img1_path.exists():
            
            img0 = cv2.imread(str(img0_path), cv2.IMREAD_UNCHANGED)
            img1 = cv2.imread(str(img1_path), cv2.IMREAD_UNCHANGED)
            
            img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
            img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

            # RIFE Padding Logic
            n, c, h, w = img0.shape
            ph = ((h - 1) // 32 + 1) * 32
            pw = ((w - 1) // 32 + 1) * 32
            padding = (0, pw - w, 0, ph - h)
            img0 = F.pad(img0, padding)
            img1 = F.pad(img1, padding)

            mid = model.inference(img0, img1)

            # REVERT PADDING
            mid_img = (mid[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            
            cv2.imwrite(str(save_path), mid_img)
            
            count += 1
            if count % 50 == 0:
                print(f"Processed {count} triplets...")

print(f"Finished successfully! Generated {count} RIFE frames.")
