# One-time setup for each new Colab session (SAM 3, frame by frame).
# Run in a notebook cell (after mounting Drive):
#   !python "/content/drive/MyDrive/video labelling/sam3 frame by frame/colab_setup.py"
import os
import subprocess
import sys

PROJECT = "/content/drive/MyDrive/video labelling"  # the shared SAM patch lives here
SAM_DIR = "/content/sam3-repo"
SAM_COMMIT = "7f473cf"  # same version as the laptop
WEIGHTS_DIR = "/content/weights"


def run(cmd):
    print(">>", cmd)
    subprocess.run(cmd, shell=True, check=True)


pip = f'"{sys.executable}" -m pip install --quiet'

# 1. PyTorch with CUDA 12.8 (needed for the G4's Blackwell GPU)
run(f"{pip} torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128")

# 2. SAM code, pinned to the laptop's version, plus our fixes
if not os.path.exists(SAM_DIR):
    run(f"git clone https://github.com/facebookresearch/sam3.git {SAM_DIR}")
    run(f"git -C {SAM_DIR} checkout {SAM_COMMIT}")
    run(f'git -C {SAM_DIR} apply "{PROJECT}/sam3_fixes.patch"')
run(f"{pip} -e {SAM_DIR}")

# 3. Extra libraries SAM needs (OpenCV 4.11 works with numpy 1.x, which SAM requires)
run(f'"{sys.executable}" -m pip uninstall --quiet -y opencv-python opencv-contrib-python')
run(f'{pip} einops pycocotools psutil "numpy>=1.26,<2" "opencv-python-headless<4.12"')

# 4. SAM 3 (not 3.1) weights from a mirror of Meta's file (same size as the official sam3.pt),
#    saved on the Colab machine, not Drive, for speed
run(
    f'"{sys.executable}" -c "from huggingface_hub import hf_hub_download; '
    f"print(hf_hub_download(repo_id='1038lab/sam3', filename='sam3.pt', local_dir='{WEIGHTS_DIR}'))\""
)

print("Setup done.")
