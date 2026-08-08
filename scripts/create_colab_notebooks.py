"""
create_colab_notebooks.py
=========================
Helper script to generate Stage 1 and Stage 2 Colab Notebooks (.ipynb) dynamically.
This bypasses the IDE's file extension restriction for directly editing .ipynb files.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

# Define Stage 1 Notebook JSON structure
stage1_nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# **DeepSentinel — Google Colab Stage 1 (Phase 1) Training**\n",
                "\n",
                "This notebook automates **Stage 1 (Phase 1) training** using the **Emotion-Space Bilinear Pooling** architecture.\n",
                "Phase 1 trains the classification heads and bilinear pooling on the precomputed feature cache (frozen encoders).\n",
                "\n",
                "### **Google Drive Folder Setup**\n",
                "Ensure your Google Drive has the following structure and file:\n",
                "```\n",
                "My Drive/\n",
                "└── THESIS_MOTHERFILE/\n",
                "    └── preprocessed/\n",
                "        └── preprocessed.zip  (Or Copy of preprocessed_all.zip)\n",
                "```"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## **Step 1: Mount Google Drive**\n",
                "Mount your Google Drive to access the preprocessed features zip file."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "from google.colab import drive\n",
                "drive.mount('/content/drive')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## **Step 2: Clone Repository & Install Dependencies**"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Change directory to safety in case of runtime restart\n",
                "%cd /content\n",
                "# Clean up any old repository folder\n",
                "!rm -rf /content/thesis\n",
                "# Clone the repository containing the training code and checkout the current branch\n",
                "!git clone -b feat/training-turnover-prep https://github.com/gjvlio/emotion-based-multimodal-deepfake-detector.git /content/thesis\n",
                "%cd /content/thesis\n",
                "\n",
                "# Install required Python dependencies\n",
                "!pip install -q transformers scikit-learn tensorboard timm pandas"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## **Step 3: Run Stage 1 Training (40 Epochs)**\n",
                "This runs `scripts/colab_stage1.py` which will extract preprocessed files from your Drive, generate training split manifests, train for **40 epochs**, and save the final checkpoint to Google Drive."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "!python scripts/colab_stage1.py"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "toc_visible": True
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Define Stage 2 Notebook JSON structure
stage2_nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# **DeepSentinel — Google Colab Stage 2 (Phase 2) Training**\n",
                "\n",
                "This notebook automates **Stage 2 (Phase 2) training** (end-to-end backbone fine-tuning) and runs the final cross-dataset benchmark evaluation on FakeAVCeleb.\n",
                "\n",
                "### **Google Drive Folder Setup**\n",
                "Ensure your Google Drive has the following structure and files:\n",
                "```\n",
                "My Drive/\n",
                "└── THESIS_MOTHERFILE/\n",
                "    ├── datasets/\n",
                "    │   ├── tracks_1_2_3_4.zip\n",
                "    │   ├── meld_raw.zip\n",
                "    │   ├── mustard.zip\n",
                "    │   └── Fakeavceleb.zip\n",
                "    └── checkpoints/\n",
                "        └── best_phase1_emotion_bilinear.pt (Generated in Stage 1)\n",
                "```"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## **Step 1: Mount Google Drive**\n",
                "Mount Google Drive to load the Stage 1 weights and dataset zip files."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "from google.colab import drive\n",
                "drive.mount('/content/drive')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## **Step 2: Clone Repository & Install Dependencies**"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Change directory to safety in case of runtime restart\n",
                "%cd /content\n",
                "# Clean up any old repository folder\n",
                "!rm -rf /content/thesis\n",
                "# Clone the repository containing the training code and checkout the current branch\n",
                "!git clone -b feat/training-turnover-prep https://github.com/gjvlio/emotion-based-multimodal-deepfake-detector.git /content/thesis\n",
                "%cd /content/thesis\n",
                "\n",
                "# Install required Python dependencies\n",
                "!pip install -q transformers scikit-learn tensorboard timm pandas"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## **Step 3: Run Stage 2 Fine-Tuning & Evaluation (40 Epochs)**\n",
                "This runs `scripts/colab_stage2.py` which will extract raw datasets, copy your Stage 1 weights, fine-tune the backbones for **40 epochs**, save the final checkpoint, and run the FakeAVCeleb out-of-domain benchmark evaluation."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "!python scripts/colab_stage2.py"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "toc_visible": True
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def save_notebook(nb_dict, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=1, ensure_ascii=False)
    print(f"Generated notebook: {filepath.relative_to(REPO_ROOT)}")

def main():
    save_notebook(stage1_nb, NOTEBOOKS_DIR / "colab_stage1_training.ipynb")
    save_notebook(stage2_nb, NOTEBOOKS_DIR / "colab_stage2_training.ipynb")
    print("Notebook generation completed successfully!")

if __name__ == "__main__":
    main()
