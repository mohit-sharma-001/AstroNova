import os
import numpy as np
import cv2
from datasets import load_dataset

def main():
    # 1. Tumhare structure ke hisaab se paths set karna (src -> dataset)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "..", "dataset")

    # Hum 3 alag directories banayenge organized testing ke liye
    sar_dir = os.path.join(dataset_dir, "SAR_Gallery")
    rgb_dir = os.path.join(dataset_dir, "RGB_Gallery")
    ir_dir = os.path.join(dataset_dir, "IR_Gallery")

    os.makedirs(sar_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(ir_dir, exist_ok=True)

    print("Downloading 50 samples from SEN12MS-CR Mirror...")
    # Naya working Hugging Face dataset link
    dataset = load_dataset("Hermanni/sen12mscr", split="train[:50]")

    def normalize_image(img_array):
        """16-bit satellite data ko stretch karke visible 8-bit banata hai"""
        p2, p98 = np.percentile(img_array, (2, 98))
        img_norm = np.clip(img_array, p2, p98)
        # Division by zero bachane ke liye
        if p98 - p2 > 0:
            img_norm = (img_norm - p2) / (p98 - p2) * 255.0
        else:
            img_norm = img_norm * 0
        return img_norm.astype(np.uint8)

    print(f"Processing and saving 50 images to {dataset_dir}...")

    # Sirf 50 images loop karni hain
    for i in range(50): 
        sample = dataset[i]
        
        # --- 1. SAR (Sentinel-1) ---
        # Naye dataset mein data bytes format mein hai, toh hum usko numpy se decode kar rahe hain
        sar_data = np.frombuffer(sample["sar"], dtype=np.float32).reshape(sample["sar_shape"])
        # SAR ki 2 channels (VV, VH) ka average lekar black & white image banana
        sar_img = np.mean(sar_data, axis=-1) 
        sar_visible = normalize_image(sar_img)
        cv2.imwrite(os.path.join(sar_dir, f"img_p{i+1}.png"), sar_visible)

        # --- 2. Sentinel-2 (Multispectral) Data ---
        # 'target' key mein cloud-free optical photo hoti hai
        s2_data = np.frombuffer(sample["target"], dtype=np.int16).reshape(sample["opt_shape"])
        
        # Dataset (Height, Width, Channels) shape mein hai (256, 256, 13)
        # B2=Blue(1), B3=Green(2), B4=Red(3), B8=NIR(7)
        blue = s2_data[:, :, 1]
        green = s2_data[:, :, 2]
        red = s2_data[:, :, 3]
        
        # --- A. RGB (Optical) ---
        rgb_img = np.stack([blue, green, red], axis=-1) 
        rgb_visible = normalize_image(rgb_img)
        cv2.imwrite(os.path.join(rgb_dir, f"img_p{i+1}.png"), rgb_visible)

        # --- B. IR (Infrared) ---
        ir_img = s2_data[:, :, 7]
        ir_visible = normalize_image(ir_img)
        # Heatmap colormap lagana taaki IR alag dikhe
        ir_colormap = cv2.applyColorMap(ir_visible, cv2.COLORMAP_HOT)
        cv2.imwrite(os.path.join(ir_dir, f"img_p{i+1}.png"), ir_colormap)

        if (i+1) % 10 == 0:
            print(f"Saved {i+1}/50 pairs...")

    print("Success! Prototype dataset ready.")

if __name__ == "__main__":
    main()