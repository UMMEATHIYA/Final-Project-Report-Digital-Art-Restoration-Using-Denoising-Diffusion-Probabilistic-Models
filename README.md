# Final-Project-Report-Digital-Art-Restoration-Using-Denoising-Diffusion-Probabilistic-Models

# 📝Summary – Digital Art Restoration Using Denoising Diffusion Probabilistic Models

This project presents a novel approach to digitally restoring damaged cultural artwork using Denoising Diffusion Probabilistic Models (DDPMs) and Latent Diffusion Models (LDMs). Focusing on the Dunhuang murals, a historic Chinese art collection dating back to the 4th century, the project aims to address degradation issues like cracks, fading, and missing sections through a deep learning-based, non-invasive restoration method.

<img width="685" height="340" alt="image" src="https://github.com/user-attachments/assets/a40bf075-a7a9-4dd0-aa0b-5495a93a0685" />

<img width="1290" height="1155" alt="image" src="https://github.com/user-attachments/assets/19851fec-a19b-46d8-bde2-ce6194e21e8d" />

The model architecture combines pixel-space DDPMs with latent-space diffusion to balance visual quality and computational efficiency. Key innovations include:

- Cross-attention mechanisms to guide restoration with contextual cues  
- A structure-preserving module to maintain edges and fine details  
- Conditioning strategies (mask, reference, and style) for stylistic consistency  

Training was performed in two stages:  
1. Base phase on undamaged murals to learn artistic patterns  
2. Task-specific phase using artificially damaged images for restoration learning

# Defective Murals

<img width="685" height="387" alt="image" src="https://github.com/user-attachments/assets/34334733-7ff8-4642-9f3a-9ccaf4a7d497" />


The model was evaluated using PSNR, SSIM, FID, and MSE scores and benchmarked against methods like PatchMatch, GAN-based inpainting, and prior DDPM variants like RePaint. Results showed that the proposed approach consistently outperformed baselines in both visual quality and structural integrity.

This work highlights the effectiveness of diffusion-based models in digital art conservation and offers a promising tool for historians, museums, and cultural preservation initiatives. Future directions include extending the approach to 3D murals, improving inference speed, and generalizing across diverse art styles.
# Authors: Umme Athiya, Anshuman Renuka Prasad

echo "📌 Starting Digital Art Restoration Pipeline Setup..."

# Step 1: Create and activate Python virtual environment
echo "🐍 Creating virtual environment 'art_env'..."
python3 -m venv art_env
source art_env/bin/activate

# Step 2: Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Step 3: Install Python dependencies
echo "📦 Installing required packages..."
pip install torch torchvision torchaudio
pip install opencv-python scikit-image matplotlib tqdm
pip install einops diffusers transformers accelerate
pip install lpips wandb

# Optional: If using custom modules or configs
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Step 4: Prepare dataset
echo "📂 Checking dataset directory..."
DATA_DIR="data/MuralDH"
if [ ! -d "$DATA_DIR" ]; then
    echo "⚠️ Dataset not found at $DATA_DIR."
    echo "📥 Please download and place the MuralDH dataset here before proceeding."
    exit 1
else
    echo "✅ MuralDH dataset found."
fi

# Step 5: Run base training (Phase 1)
echo "🎨 Running base training phase..."
python train_base.py --data_dir "$DATA_DIR" --epochs 50 --save_dir "checkpoints/base"

# Step 6: Run task-specific restoration training (Phase 2)
echo "🩹 Running task-specific learning phase on damaged images..."
python train_restoration.py --data_dir "$DATA_DIR" --mask_dir "data/masks" --epochs 100 --pretrained "checkpoints/base/model.pt" --save_dir "checkpoints/restoration"

# Step 7: Run inference / restoration
echo "🖼️ Performing image restoration..."
python restore.py --input_dir "samples/damaged" --output_dir "samples/restored" --model_path "checkpoints/restoration/model.pt"

# Step 8: Evaluate results
echo "📊 Evaluating restorations..."
python evaluate.py --restored_dir "samples/restored" --ground_truth_dir "samples/ground_truth"

echo "✅ Restoration pipeline complete."
