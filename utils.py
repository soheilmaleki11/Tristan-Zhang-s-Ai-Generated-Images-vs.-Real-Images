import torch
import torchvision
from lightning.pytorch import LightningModule
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from tqdm.auto import tqdm



def lower_img_size(root: str, new_root: str, format: str = "JPEG", quality: int = 85, optimize: bool = True) -> None:
    """
    TRISTAN ZHANG's ai-generated-images-vs-real-images kaggle dataset is a heavy dataset (around 50GB)
    and therefore using the raw data results in significant disk I/O overhead. This function is used to largely reduce
    the dataset size while preserving visual semantics and aspect ratio using Bicubic Convolution Interpolation.

    Parameters
    ----------
    root : str
        Root directory of the original dataset (~50GB).
    new_root : str
        Root directory where the resized dataset will be stored.
    format : str, optional
        Output image format (default: "JPEG").
    quality : int, optional
        JPEG quality factor (default: 85).
    optimize : bool, optional
        Whether to enable encoder optimization (default: True).
    """
    # Run the resizing process only if the target directory does not already exist
    # Make the new root directory and subdirectories
    if not os.path.isdir(new_root):
        os.makedirs(os.path.join(new_root, 'train', 'fake'), exist_ok=True)
        os.makedirs(os.path.join(new_root, 'train', 'real'), exist_ok=True)
        os.makedirs(os.path.join(new_root, 'test', 'real'), exist_ok=True)
        os.makedirs(os.path.join(new_root, 'test', 'fake'), exist_ok=True)
    
        splits = ['train', 'test']
        classes = ['fake', 'real']
        for split in tqdm(splits, desc=f'train and test Splits', leave=True):
            for class_name in tqdm(classes, desc=f'Classes', leave=False):
                file_names = [name for name in os.listdir(os.path.join(root, split, class_name))]
                for name in tqdm(file_names, desc='Images', leave=False):
                    try:
                        # Open and convert the image to RGB
                        img = Image.open(os.path.join(root, split, class_name, name)).convert("RGB")

                        # Rescale the image so that the largest dimension is 256. Scale the other dimension accordingly.
                        # Use bicubic convolution interpolation for high-quality downsampling.
                        w, h = img.size
                        scale = 256 / max(w, h)
                        if scale < 1:
                            img = img.resize(
                                (max(int(round(w * scale, 0)), 1), max(int(round(h * scale, 0)), 1)),
                                resample=Image.Resampling.BICUBIC
                            )

                        # Save the resized image with the specified format, quality and optimization if needed
                        img.save(
                            os.path.join(new_root, split, class_name, name),
                            format=format,
                            quality=quality,
                            optimize=optimize
                        )
                    except Exception as e:
                        # Log any unexpected errors and continue processing
                        print(f'Unexpected error happened:\n{e}')

    else:
        # Inform the user that resizing is skipped since the target directory already exists
        print(f"New root path '{new_root}' already exists.")



def sample_images_labels_predictions_visualization(trained_model: LightningModule, data_root: str, split: str,
                                                    transform: torchvision.transforms.Compose, device: str = 'cpu') -> None:
    """
    This function samples 6 images from each class ('fake' or 'real'), predicts their class and draws the images with their corresponding
    labels and predictions in matplotlib subplots format.

    Parameters
    ----------
    trained_model : LightningModule
        The trained model.
    data_root : str
        The root path of the dataset.
    split : str
        The split ('train' or 'test') that images are randomely drawn from.
    transform : torchvision.transforms.Compose
        Transforms to apply on the dataset before giving them to the trained model.
    device : str
        The device to run inference on (default: 'cpu').
    """

    # Set the model to evaluation mode and move it to the device chosen.
    trained_model.eval().to(device)
    
    # Sample 6 images from each class to draw.
    data_to_draw = {}
    classes = ['fake', 'real']
    for i, class_name in enumerate(classes):
        data_to_draw[i] = np.random.choice(os.listdir(os.path.join(data_root, split, class_name)), size=6, replace=False)

    # Plot the images, their corresponding labels and predictions
    fig, axes = plt.subplots(4, 3, figsize=(10, 8), layout='constrained')
    for i in range(12):
        if i < 6:
            image_names = data_to_draw[0]
            label = classes[0]
            image_paths = [os.path.join(data_root, split, label, image_name) for image_name in image_names]
            img = Image.open(image_paths[i]).convert('RGB')
            axes.flat[i].imshow(img)
            axes.flat[i].axis('off')

            # Rescale the image so that the largest dimension is 256. Scale the other dimension accordingly.
            # Use bicubic convolution interpolation for high-quality downsampling. This transformation is done so that the inference
            # data has the same distribution as the resized dataset.
            w, h = img.size
            scale = 256 / max(w, h)
            if scale < 1:
                img = img.resize(
                    (max(int(round(w * scale, 0)), 1), max(int(round(h * scale, 0)), 1)),
                    resample=Image.Resampling.BICUBIC
                )
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                img_tensor = img_tensor.to(device)
                logits = trained_model(img_tensor)
                pred = logits.argmax(1).item()

            if pred == 0:
                axes.flat[i].set_title(f'Label: {label}\nPrediction: {classes[pred]}', color='green')
            elif pred == 1:
                axes.flat[i].set_title(f'Label: {label}\nPrediction: {classes[pred]}', color='red')
            else:
                raise ValueError(f'Prediction is {pred} and it is not 0 or 1')

        else:
            j = i - 6
            image_names = data_to_draw[1]
            label = classes[1]
            image_paths = [os.path.join(data_root, split, label, image_name) for image_name in image_names]
            img = Image.open(image_paths[j]).convert('RGB')
            axes.flat[i].imshow(img)
            axes.flat[i].axis('off')

            w, h = img.size
            scale = 256 / max(w, h)
            if scale < 1:
                img = img.resize(
                    (max(int(round(w * scale, 0)), 1), max(int(round(h * scale, 0)), 1)),
                    resample=Image.Resampling.BICUBIC
                )
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                img_tensor = img_tensor.to(device)
                logits = trained_model(img_tensor)
                pred = logits.argmax(1).item()

            if pred == 0:
                axes.flat[i].set_title(f'Label: {label}\nPrediction: {classes[pred]}', color='red')
            elif pred == 1:
                axes.flat[i].set_title(f'Label: {label}\nPrediction: {classes[pred]}', color='green')
            else:
                raise ValueError(f'Prediction is {pred} and it is not 0 or 1')
            
    plt.show()
    plt.close('all')



def show_and_predict_custom_img(img_path: str, trained_model: LightningModule, transform: torchvision.transforms.Compose,
                                    label: str | None = None, device: str = 'cpu') -> int:
    """
    This function loads a custom image, transforms it and predict whether it is generated by AI (fake) or real.

    Parameters
    ----------
    img_path : str
        The path to the custom image.
    trained_model : LightningModule
        The trained model object.
    transform : torchvision.transforms.Compose
        The transform applied on the custom image.
    label : str, optional
        The truth label of the custom image (default: None).
    device : str
        The device to run inference on (default: 'cpu').

    Returns
    -------
    prediction : int
        The class id predicted for the custom image by the trained model object.
    """

    classes = ['fake', 'real']

    # Show the image and remove the axis for better visualization.
    img = Image.open(img_path).convert('RGB')
    figure, ax = plt.subplots(figsize=(6, 5), layout='constrained')
    ax.imshow(img)
    ax.axis('off')

    w, h = img.size
    scale = 256 / max(w, h)
    if scale < 1:
        img = img.resize(
            (max(int(round(w * scale, 0)), 1), max(int(round(h * scale, 0)), 1)),
            resample=Image.Resampling.BICUBIC
        )
    
    # Transform and tensorize the image, add batch dimension and move it to the specified device.
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Move the model into the specified device and change it into evaluation mode.
    trained_model.eval().to(device)

    # Make prediction on image tensor using the trained model. Disable gradients calculations for efficiency.
    with torch.no_grad():
        logits = trained_model(img_tensor)
        prediction = logits.argmax(1).item()

    # Specify color for better visability of label and prediction.
    if label is not None:
        if classes[prediction] == label:
            color = 'green'
        else:
            color = 'red'
        ax.set_title(f'Label: {label.title()}\nPrediction: {classes[prediction].title()}', color=color)
    else:
        ax.set_title(f'Prediction: {classes[prediction].title()}')

    plt.show()
    plt.close('all')

    return prediction