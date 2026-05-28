import torch
from lightning.pytorch import LightningDataModule
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import gc
import csv
from IPython.display import clear_output
from tqdm.auto import tqdm



class AIvsRealDataset(LightningDataModule):
    def __init__(self, data_root: str, train_batch_size: int, val_batch_size: int, num_workers: int = 0, persistent_workers: bool = False,
                    pin_memory: bool = False, prefetch_factor: int = 2, train_transform: torchvision.transforms.Compose | None = None,
                    val_transform: torchvision.transforms.Compose | None = None, train_dir: str = 'train', val_dir: str = 'test') -> None:
        """
        PyTorch Lightning DataModule for AI-generated versus real image classification.

        This module handles the setup of training and validation datasets, dataloaders,
        and prepares transforms for image augmentation and preprocessing.

        Parameters
        ----------
        data_root : str
            Root directory path of the dataset on local storage.
        train_batch_size : int
            Batch size used for the training dataloader.
        val_batch_size : int
            Batch size used for the validation dataloader.
        num_workers : int, optional
            Number of subprocesses used for parallel image loading (default: 0).
        persistent_workers : bool, optional
            If True, the data loader will not shut down the worker processes after a dataset has been consumed once (default: False).
        pin_memory : bool, optional
            If True, moves data to pinned (page-locked) memory for faster GPU transfer (default: False).
        prefetch_factor : int, optional
            Number of batches preloaded by each worker (default: 2).
        train_transform : torchvision.transforms.Compose | None, optional
            Transformations applied to training images (default: None).
        val_transform : torchvision.transforms.Compose | None, optional
            Transformations applied to validation images (default: None).
        train_dir : str, optional
            Subdirectory name for the training dataset (default: 'train').
        val_dir : str, optional
            Subdirectory name for the validation dataset (default: 'test').
        """
        # Call parent class constructor (LightningDataModule)
        super().__init__()

        # Store input arguments as instance attributes
        self.root = data_root
        self.train_dir = os.path.join(self.root, train_dir)
        self.val_dir = os.path.join(self.root, val_dir)
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor



    def setup(self, stage: str | None = None) -> None:
        """
        This method sets up the train and validation datasets and stores the class names and number of classes.

        Parameters
        ----------
        stage : str | None, optional
            The stage of the model. This parameter is not used in this implementation (default: None).
        """
        # Setup the train dataset
        self.train_dataset = ImageFolder(self.train_dir, transform=self.train_transform)

        # Setup the validation dataset
        self.val_dataset = ImageFolder(self.val_dir, transform=self.val_transform)

        # Save class names for reference and visualization
        self.class_names = self.train_dataset.classes

        # Count total number of classes
        self.num_classes = len(self.class_names)



    def train_dataloader(self) -> DataLoader:
        """
        This method sets up the train dataloader using the arguments given in the class definition.

        Returns
        -------
        torch.utils.data.dataloader.DataLoader object
            The train dataloader object
        """
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers,
                            persistent_workers=self.persistent_workers, pin_memory=self.pin_memory, prefetch_factor=self.prefetch_factor)



    def val_dataloader(self) -> DataLoader:
        """
        This method sets up the validation dataloader using the arguments given in the class definition.

        Returns
        -------
        torch.utils.data.dataloader.DataLoader object
            The validation dataloader object
        """
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers,
                            persistent_workers=self.persistent_workers, pin_memory=self.pin_memory, prefetch_factor=self.prefetch_factor)



def test_data_module(data_root: str, train_batch_size: int, val_batch_size: int, train_dir: str = 'train', val_dir: str = 'test',
                     num_workers: int = 0, pin_memory: bool = False, prefetch_factor: int = 2) -> None:
    """
    This function is used to test the lightning data module and visualize samples of ai-generated-images-vs-real-images dataset.

    Parameters
    ----------
    data_root : str
        The root path of the dataset on local storage.
    train_batch_size : int
        Batch size of the train dataloader.
    val_batch_size : int
        Batch size of the validation dataloader.
    train_dir : str, optional
        Name of train directory (default: 'train').
    val_dir : str, optional
        Name of validation directory (default: 'test').
    num_workers : int, optional
        Number of subprocesses used for parallel image loading (default: 0).
    pin_memory : bool, optional
        If True, moves data to pinned (page-locked) memory for faster GPU transfer (default: False).
    prefetch_factor : int, optional
        Number of batches preloaded by each worker (default: 2).
    """
    # Define lightweight transforms for quick sanity-checking
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Instantiate a test lightning data module object with given arguments
    test_data = AIvsRealDataset(
        data_root=data_root,
        train_dir=train_dir,
        val_dir=val_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor
    )

    # Initialize datasets (required before creating dataloaders)
    test_data.setup()

    # Create dataloaders for sanity checks
    test_train_loader = test_data.train_dataloader()
    test_val_loader = test_data.val_dataloader()

    print(f'Number of training instances: {len(test_data.train_dataset)}')
    print(f'Number of validation instances: {len(test_data.val_dataset)}')
    print()
    print(f'Shape of images of a sample batch from training loader: {next(iter(test_train_loader))[0].shape}')
    print(f'Shape of labels of a sample batch from training loader: {next(iter(test_train_loader))[1].shape}')
    print(f'Shape of images of a sample batch from val loader: {next(iter(test_val_loader))[0].shape}')
    print(f'Shape of labels of a sample batch from val loader: {next(iter(test_val_loader))[1].shape}')

    # Visualize a small batch of training samples for sanity checking
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 8), layout='constrained')
    batch = next(iter(test_train_loader))
    for i, ax in enumerate(axes.flat):
        img = batch[0][i]
        img = img.permute(1, 2, 0)
        label = batch[1][i]
        ax.imshow(img)
        ax.set_title(f'{test_data.class_names[label]}')
        ax.axis('off')
    fig.suptitle('Sample of train images')
    plt.show()
    plt.close('all')

    # Explicitly delete large objects to free memory in interactive sessions
    del train_transform, val_transform, test_data, test_train_loader, test_val_loader, axes, fig

    # Empty GPU cache
    torch.cuda.empty_cache()
    
    # Garbage collection
    gc.collect()



def study_dataloader_pipeline(num_workers_candidates: list, train_batch_size_candidates: list, warm_up: int, epochs: int,
                                device: str = 'cpu', transform: torchvision.transforms.Compose | None = None) -> dict:
    """
    Tests dataset I/O, transformations, augmentations, tensorizations and movement to the appropriate device and returns a fine
    combination of settings to use. Prefetch factor is always set to 2 for this project. A csv file is created on disk, too.

    Parameters
    ----------
    num_workers_candidates : list
        A python list containing the candidate values of number of workers to test.
    train_batch_size_candidates : list
        A python list containing the candidate values of batch sizes to test.
    warm_up : int
        Number of epochs to warm-up the process.
    epochs : int
        Number of epochs to run the test and average the results over for each setting.
    device : str, Optional
        The device to move the final tensors to (default: 'cpu').
    transform : torchvision.transforms.Compose | None, Optional
        The transformations to apply to the dataset (default: None).

    Returns
    -------
    results : dict
        A python dictionary containing the best settings found.
    """
    # Load csv file containing optimal data loader pipeline settings if it already exists in storage.
    if os.path.exists('dataloader_pipeline_optimal_settings.csv'):
        print('A csv file already exists containing the optimal data loader pipeline settings.')
        print("Please use 'load_dataloader_optimal_settings()' function to load the csv data into a python dictionary and proceed.")
        return
    
    # Candidate values for different settings and options.
    data_preprocessing_hyperparameters = {
        'num_workers': num_workers_candidates,
        'persistent_workers': [True], # In this project, we let 'persistent_workers' only to be True.
        'pin_memory': [False, True],
        'train_batch_size': train_batch_size_candidates,
    }

    # Dictionaries for storing the test results of the candidates.
    num_workers_results = {}
    pin_memory_results = {}
    persistent_workers_results = {}
    train_batch_size_results = {}

    # Default values to use for dataloader pipeline.
    persistent_workers_default = False
    pin_memory_default = False
    train_batch_size_default = 32

    # Initial setups.
    test_train_dataset = ImageFolder('data_resized', transform=transform)

    # Number of workers tests.
    for num_workers in data_preprocessing_hyperparameters['num_workers']:
        # Instantiate dataloader.
        test_train_loader = DataLoader(
            test_train_dataset,
            batch_size=train_batch_size_default,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers_default,
            pin_memory=pin_memory_default,
            prefetch_factor=2,
        )

        print('Warmup Round...')
        for epoch in range(warm_up):
            progress_bar = tqdm(test_train_loader, desc=f"[WARMUP] 'num_workers' = {num_workers} | Epoch {epoch + 1}/{warm_up}",
                                    leave=False)

            # Get batches of data and transform them and move them to the chosen device. Used tqdm for progress bars.
            for feats, labels in progress_bar:
                feats, labels = feats.to(device), labels.to(device)
        print('Warmup finished.')

        # Make sure GPU (if used) is exhausted all its works completely before logging the start time.
        torch.cuda.synchronize()
        
        # Log the start of the test. Use 'perf_counter()' instead of 'time.time()' for more precision.
        start_time = time.perf_counter()
        for epoch in range(epochs):
            progress_bar = tqdm(test_train_loader, desc=f"[Actual] 'num_workers' = {num_workers} | Epoch {epoch + 1}/{epochs}",
                                    leave=False)

            # Get batches of data and transform them and move them to the chosen device. Used tqdm for progress bars.
            for feats, labels in progress_bar:
                feats, labels = feats.to(device), labels.to(device)
        
        # Make sure GPU (if used) is exhausted all its works completely before logging the duration.
        torch.cuda.synchronize()

        # Log the average epoch time resulted from tests.
        num_workers_results[num_workers] = (time.perf_counter() - start_time) / epochs

        print(f"'num_workers' = {num_workers} took {num_workers_results[num_workers]:.2f} seconds.")
        
        # Wait 5 seconds for the previous print statement to show for 5 seconds before clearing it.
        time.sleep(5)
        
        # Clear the terminal outputs
        clear_output()

        # Delete the dataloader for efficiency and test isolations.
        del test_train_loader
        
        # Clear the GPU (if used) cache for efficiency and test isolations.
        torch.cuda.empty_cache()
        
        # Garbage collection for efficiency and test isolations.
        gc.collect()

    # Best result index.
    min_idx = np.array(list(num_workers_results.values())).argmin()

    # Choose the best option.
    best_num_workers = np.array(list(num_workers_results.keys()))[min_idx]

    print(f"Best 'num_workers' = {best_num_workers} with average time per epoch = {num_workers_results[best_num_workers]:.3f} seconds.")
    time.sleep(5)
    clear_output()



    # Persistent workers tests.
    for persistent_workers in data_preprocessing_hyperparameters['persistent_workers']:
        test_train_loader = DataLoader(
            test_train_dataset,
            batch_size=train_batch_size_default,
            shuffle=True,
            num_workers=best_num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory_default,
            prefetch_factor=2,
        )

        print('Warmup Round...')
        for epoch in range(warm_up):
            progress_bar = tqdm(test_train_loader,
                                    desc=f"[WARMUP] 'persistent_workers' = {persistent_workers} | Epoch {epoch + 1}/{warm_up}",
                                    leave=False)
            
            for feats, labels in progress_bar:
                feats, labels = feats.to(device), labels.to(device)
        print('Warmup finished.')

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for epoch in range(epochs):
            progress_bar = tqdm(test_train_loader, desc=f"[Actual] 'persistent_workers' = {persistent_workers} | Epoch {epoch + 1}/{epochs}",
                                    leave=False)
            
            for feats, labels in progress_bar:
                feats, labels = feats.to(device), labels.to(device)
        torch.cuda.synchronize()
        persistent_workers_results[persistent_workers] = (time.perf_counter() - start_time) / epochs

        print(f"'persistent_workers' = {persistent_workers} took {persistent_workers_results[persistent_workers]:.2f} seconds.")
        time.sleep(5)
        clear_output()

        del test_train_loader
        torch.cuda.empty_cache()
        gc.collect()

    min_idx = np.array(list(persistent_workers_results.values())).argmin()
    best_persistent_workers = np.array(list(persistent_workers_results.keys()))[min_idx]
    print(
        f"Best 'persistent_workers' = {best_persistent_workers} with average time per"
        f" epoch = {persistent_workers_results[best_persistent_workers]:.3f} seconds."
    )
    time.sleep(5)
    clear_output()



    # Pin memory tests.
    for pin_memory in data_preprocessing_hyperparameters['pin_memory']:
        test_train_loader = DataLoader(
            test_train_dataset,
            batch_size=train_batch_size_default,
            shuffle=True,
            num_workers=best_num_workers,
            persistent_workers=best_persistent_workers,
            pin_memory=pin_memory,
            prefetch_factor=2,
        )

        print('Warmup Round...')
        for epoch in range(warm_up):
            progress_bar = tqdm(test_train_loader, desc=f"[WARMUP] 'pin_memory' = {pin_memory} | Epoch {epoch + 1}/{warm_up}", leave=False)
            for feats, labels in progress_bar:
                feats, labels = feats.to(device), labels.to(device)
        print('Warmup finished.')

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for epoch in range(epochs):
            progress_bar = tqdm(test_train_loader, desc=f"[Actual] 'pin_memory' = {pin_memory} | Epoch {epoch + 1}/{epochs}", leave=False)
            for feats, labels in progress_bar:
                feats, labels = feats.to(device), labels.to(device)
        torch.cuda.synchronize()
        pin_memory_results[pin_memory] = (time.perf_counter() - start_time) / epochs

        print(f"'pin_memory' = {pin_memory} took {pin_memory_results[pin_memory]:.2f} seconds.")
        time.sleep(5)
        clear_output()

        del test_train_loader
        torch.cuda.empty_cache()
        gc.collect()

    min_idx = np.array(list(pin_memory_results.values())).argmin()
    best_pin_memory = np.array(list(pin_memory_results.keys()))[min_idx]
    print(f"Best 'pin_memory' = {best_pin_memory} with average time per epoch = {pin_memory_results[best_pin_memory]:.3f} seconds.")
    time.sleep(5)
    clear_output()



    # Train barch size tests.
    for train_batch_size in data_preprocessing_hyperparameters['train_batch_size']:
        test_train_loader = DataLoader(
            test_train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=best_num_workers,
            persistent_workers=best_persistent_workers,
            pin_memory=best_pin_memory,
            prefetch_factor=2,
        )

        print('Warmup Round...')
        for epoch in range(warm_up):
            progress_bar = tqdm(test_train_loader, desc=f"[WARMUP] 'train_batch_size' = {train_batch_size} | Epoch {epoch + 1}/{warm_up}",
                                    leave=False)
            
            for feats, labels in progress_bar:
                feats, labels = feats.to(device), labels.to(device)
        print('Warmup finished.')

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for epoch in range(epochs):
            progress_bar = tqdm(test_train_loader, desc=f"[Actual] 'train_batch_size' = {train_batch_size} | Epoch {epoch + 1}/{epochs}",
                                    leave=False)
            
            for feats, labels in progress_bar:
                feats, labels = feats.to(device), labels.to(device)
        torch.cuda.synchronize()
        train_batch_size_results[train_batch_size] = (time.perf_counter() - start_time) / epochs

        print(f"'train_batch_size' = {train_batch_size} took {train_batch_size_results[train_batch_size]:.2f} seconds.")
        time.sleep(5)
        clear_output()

        del test_train_loader
        torch.cuda.empty_cache()
        gc.collect()

    min_idx = np.array(list(train_batch_size_results.values())).argmin()
    best_train_batch_size = np.array(list(train_batch_size_results.keys()))[min_idx]
    print(
        f"Best 'train_batch_size' = {best_train_batch_size} with average time per "
        f"epoch = {train_batch_size_results[best_train_batch_size]:.3f} seconds."
    )
    time.sleep(5)
    clear_output()

    # Save a csv file on disk with the best values of the settings.
    with open('dataloader_pipeline_optimal_settings.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Option', 'Value'])
        
        writer.writerow(['num_workers', best_num_workers])
        writer.writerow(['persistent_workers', best_persistent_workers])
        writer.writerow(['pin_memory', best_pin_memory])
        writer.writerow(['train_batch_size', best_train_batch_size])

    # Plot the results for visibility.
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), layout='constrained')
    axes[0, 0].plot(list(num_workers_results.keys()), list(num_workers_results.values()), marker='o', markersize=6, linestyle='--',
                        c='blue')
    axes[0, 0].axis('on')
    axes[0, 0].set_xlabel('Number of Workers')
    axes[0, 0].set_ylabel('Average Time per Epoch (seconds)')

    axes[0, 1].plot(list(pin_memory_results.keys()), list(pin_memory_results.values()), marker='o', markersize=6, linestyle='--',
                        c='blue')
    axes[0, 1].axis('on')
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_xticklabels(['False', 'True'])
    axes[0, 1].set_xlabel('Pin Memory')
    axes[0, 1].set_ylabel('Average Time per Epoch (seconds)')

    axes[1, 0].plot(list(persistent_workers_results.keys()), list(persistent_workers_results.values()), marker='o', markersize=6,
                        linestyle='--', c='blue')
    axes[1, 0].axis('on')
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_xticklabels(['False', 'True'])
    axes[1, 0].set_xlabel('Persistent Workers')
    axes[1, 0].set_ylabel('Average Time per Epoch (seconds)')

    axes[1, 1].plot(list(train_batch_size_results.keys()), list(train_batch_size_results.values()), marker='o', markersize=6,
                        linestyle='--', c='blue')
    axes[1, 1].axis('on')
    axes[1, 1].set_xlabel('Train Batch Size')
    axes[1, 1].set_ylabel('Average Time per Epoch (seconds)')

    fig.suptitle('Dataloader Pipeline Settings')
    plt.show()
    plt.close('all')

    # Return the final results as a python dictionary.
    results = {
        'num_workers': best_num_workers,
        'persistent_workers': best_persistent_workers,
        'pin_memory': best_pin_memory,
        'train_batch_size': best_train_batch_size,
    }

    return results



def load_dataloader_optimal_settings() -> dict:
    """
    Loads a python dictionary containing optimal data loader pipeline settings.

    Returns
    -------
    dataloader_pipeline_settings
        A python dictionary containing optimal data loader pipeline settings.
    """
    dataloader_pipeline_settings = {}
    with open('dataloader_pipeline_optimal_settings.csv', mode='r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            try:
                data = int(row[1])
                dataloader_pipeline_settings[row[0]] = data
            except:
                if row[1].upper() == 'TRUE':
                    data = True
                    dataloader_pipeline_settings[row[0]] = data
                elif row[1].upper() == 'FALSE':
                    data = False
                    dataloader_pipeline_settings[row[0]] = data
                else:
                    raise ValueError(f"'{row[0]}' is boolean type but has value '{row[1]}' which is other than 'True' or 'False'")

    return dataloader_pipeline_settings