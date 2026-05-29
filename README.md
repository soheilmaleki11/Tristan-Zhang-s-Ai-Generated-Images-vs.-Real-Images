# Tristan Zhang's AI Generated Images vs. Real Images Binary Classification

This project implements a deep learning model based on ResNet-50 for classifying [Tristan Zhang's AI generated images vs real images](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images).

In this project, at first we convert the dataset into lower-resolution and lower-dimension images with the same context using bicubic convolutional interpolation for faster I/O. Then, we use `lightning.pytorch.LightningDataModule` and `lightning.pytorch.LightningModule` to create the dataset and model objects, respectively. Later, we search data loader pipeline settings and hyperparameters space and find a good combination of data loader pipeline settings and hyperparameter values for our problem. Finally, we train a new
model using optimal data loader pipeline settings and hyperparameters values using 2-stage fine-tuning of ResNet-50 pretrained on ImageNet, print metrics, plot visualizations and predict sample images from the dataset.

## Model Architecture

- **Backbone:** ResNet-50 with default weights
- **Image Size:** 224x224
- **Framework:** PyTorch Lightning
- **Loss:** Categorical Crossentropy
- **Optimizer:** AdamW

## Dataset

- **Source:** [ai-generated-images-vs-real-images](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images)
- **Number of Images:** 60,000
- **Classes:** 2
- **Train/Test Split:** 80/20

## Installation

```bash
pip install -r requirements.txt
```

## Training/Inference

Run the `tristan_zhang_ai_generated_images_vs_real_images.ipynb` jupyter notebook.

## Hardware

- **CPU:** Intel CoreI7 13650H
- **GPU:** GeForce RTX-4060 8GB VRAM

## Results

The model was trained for 10 epochs using transfer learning on ResNet-50.

Across multiple runs:

- **Final Train Accuracy:** ~93%-95%
- **Final Validation Accuracy:** ~90%

Performance may vary slightly due to random initialization and stochastic training.

## Project Structure

```
├── inference/
│   ├── real/
│   └── fake/
│
├── dataset.py
├── model.py
├── utils.py
├── tristan_zhang_ai_generated_images_vs_real_images.ipynb
├── requirements.txt
└── README.md
```

## References
- [Deep Residual Learning for Image Recognition by Kaiming He et al.](https://arxiv.org/abs/1512.03385)
