from lightning.pytorch import LightningDataModule, LightningModule, Trainer
import torch
import torchvision
import torchvision.models as tv_models
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import Accuracy, ConfusionMatrix
from torchvision import transforms
import matplotlib.pyplot as plt
import csv
from dataset import *



def params_group(model: nn.Module, weight_decay: float) -> list:
    """
    This function groups model's parameters into two seperate groups. One group accepts weight decay and other does not.
    Biases and Batch Normalization parameters do not need weight decay. Rest need weight decay.

    Parameters
    ----------
    model : nn.Module
        Deep Learning model being used.
    weight_decay : float
        The rate of weight decay for model parameters.

    Returns
    -------
    param_groups : list
        A list of python dictionaries, each having 'params' and 'weight_decay' as their keys; one with 'weight_decay'=weight_decay and
        other with 'weight_decay'=0.0.
    """
    # Instantiate two python lists for dividing model parameters into weight decay and no weight decay parameters
    decay, no_decay = [], []
    
    # Batch Normalization parameters and biases (1-dimensional) are not decayed. Rest are decayed
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d):
            for params in module.parameters(recurse=False):
              no_decay.append(params)
        else:
            for params in module.parameters(recurse=False):
              if params.ndim == 1:
                  no_decay.append(params)
              else:
                  decay.append(params)

    # Group the parameters by decay and no decay and return them as a python list
    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return param_groups



class AIvsRealModel(LightningModule):
  """
  This class contains the logic for the TRISTAN ZHANG's ai-generated-images-vs-real-images kaggle dataset binary classification task.

  This class handles saving hyperparameters of the model, loading ResNet-50 pretrained model, adjusting its fc layer to match the problem
  at hand and use 2-stage strategy to fine-tune the model.
  
  Parameters
  ----------
  learning_rate_stage1 : float
    The learning rate for stage 1 of fine-tuning the model.
  learning_rate_stage2 : float
    The learning rate for stage 2 of fine-tuning the model.
  weight_decay_rate_stage1 : float
    The weight decay rate of stage 1 of fine-tuning the model.
  weight_decay_rate_stage2 : float
    The weight decay rate of stage 2 of fine-tuning the model.
  num_classes : int
    Number of classes.
  num_epochs : int
    Number of epochs to train the model.
  weight_path : str, optional
    The path to the ResNet-50 pretrained weights on local storage (default: None).
  """
  def __init__(self, learning_rate_stage1: float, learning_rate_stage2: float, weight_decay_rate_stage1: float,
                weight_decay_rate_stage2: float, num_epochs: int, num_classes: int, weight_path: str = None) -> None:
    # Run __init__() of the main class
    super().__init__()
    
    # Save __init__() arguments inside module's object attribute 'hparams'
    self.save_hyperparameters()

    # If 'weight_path' is provided, the model's weights are loaded locally; else, it is loaded online.
    if weight_path is not None:
      weights = torch.load(weight_path, map_location='cpu')
      self.model = tv_models.resnet50(weights=None)
      self.model.load_state_dict(weights)
    else:
      weights = tv_models.ResNet50_Weights.DEFAULT
      self.model = tv_models.resnet50(weights=weights)

    # Make final fc layer to be specific for the problem at hand.
    input_features = self.model.fc.in_features
    new_fc = nn.Linear(input_features, num_classes)
    self.model.fc = new_fc

    # Only let fc parameters be trained.
    for params in self.model.parameters():
      params.requires_grad = False
    for params in self.model.fc.parameters():
      params.requires_grad = True

    # Instantiate stage, loss function, train and validation set accuracy metrics and confusion matrix object.
    self.stage = 1
    self.loss_fn = nn.CrossEntropyLoss()
    self.train_acc = Accuracy(task='multiclass', num_classes=num_classes, average='micro')
    self.val_acc = Accuracy(task='multiclass', num_classes=num_classes, average='micro')
    self.cm = ConfusionMatrix(task='multiclass', num_classes=num_classes)



  def fix_layers_second_stage(self) -> None:
    """
    This method is used for the stage 2 of fine-tuning the model. It will make layer4 and fc parameters trainable and all other parameters
    non-trainable.
    """
    # Make only fc and layer4 parameters trainable, other parameters non-trainable.
    for params in self.model.parameters():
      params.requires_grad = False
    for params in self.model.fc.parameters():
      params.requires_grad = True
    for params in self.model.layer4.parameters():
      params.requires_grad = True



  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    This method runs the 'x' input tensor through the model and return the logits tensor.

    Parameters
    ----------
    x : torch.Tensor
      Input tensor fed to the model.
    
    Returns
    -------
    logits : torch.Tensor
      Output logits of the model.
    """
    logits = self.model(x)
    return logits



  def training_step(self, batch: tuple, batch_idx: int = None) -> torch.Tensor:
    """
    This method runs one training step on the given batch, logs train loss and accuracy on epoch end and returns the loss.

    Parameters
    ----------
    batch : tuple
      Input batch for this step of training the model.
    batch_idx : int
      Index of the current batch. This argument is not used in this implementation (default: None).

    Returns
    -------
    loss : torch.Tensor
      Loss value of the current training step.
    """
    imgs, labels = batch

    logits = self(imgs)
    loss = self.loss_fn(logits, labels)

    self.train_acc(logits, labels)

    self.log_dict({
      'train_loss': loss,
      'train_acc': self.train_acc
    }, prog_bar=True, on_step=False, on_epoch=True)

    return loss



  def validation_step(self, batch: tuple, batch_idx: int = None) -> torch.Tensor:
    """
    This method runs one validation step on the given batch, logs validation loss and accuracy on epoch end and returns the loss.
    On final epoch, this method calculates confusion matrix.

    Parameters
    ----------
    batch : tuple
      Input batch for this step of validating the model.
    batch_idx : int
      Index of the current batch. This argument is not used in this implementation (default: None).

    Returns
    -------
    loss : torch.Tensor
      Loss value of the current validation step.
    """
    imgs, labels = batch

    logits = self(imgs)
    loss = self.loss_fn(logits, labels)

    self.val_acc(logits, labels)

    self.log_dict({
      'val_loss': loss,
      'val_acc': self.val_acc
    }, prog_bar=True, on_step=False, on_epoch=True)

    # Calculate confusion matrix on final epoch
    if self.current_epoch == self.trainer.max_epochs - 1:
      preds = logits.argmax(1)
      self.cm(preds, labels)

    return loss



  def on_train_epoch_start(self) -> None:
    """
    This method monitors the stage and current epoch of the model in each epoch start. If the current stage is stage 1 and 70% or more
    epochs have passed, this method automatically switch the model to stage 2 of fine-tuning, makes layer4 and fc parameters trainable
    and sets the new learning rate and weight decay.
    """
    # Check if 70% or more epochs have passed, if so, switch to stage 2 of fine-tuning the model.
    if self.stage == 1 and self.current_epoch >= int(self.hparams.num_epochs * 0.7):
      print('Switching to stage 2 of fine-tuning the model...')
      self.stage = 2
      self.fix_layers_second_stage()
      for group in self.trainer.optimizers[0].param_groups:
        group['lr'] = self.hparams.learning_rate_stage2
        if group['weight_decay'] != 0.0:
          group['weight_decay'] = self.hparams.weight_decay_rate_stage2
      print('Switching Complete. Continuing Training...')



  def on_validation_end(self) -> None:
    """
    This method runs only on final epoch of the validation. It calculates the final confusion matrix, stores it inside module's
    'final_cm' attribute and reset the metric afterwards.
    """
    # Compute the final confusion matrix on final epoch, store it and reset the metric afterward.
    if self.current_epoch == self.trainer.max_epochs - 1:
      self.final_cm = self.cm.compute().detach().cpu().numpy()
      self.cm.reset()



  def configure_optimizers(self) -> dict:
    """
    This method returns a python dictionary containing optimizer and learning rate scheduler
    monitoring 'val_acc'.

    Returns
    -------
    optimization_dict : dict
      A python dictionary containing two keys. 'optimizer' stores the optimizer and 'lr_scheduler' contains another dictionary, with two
      keys; 'scheduler' which stores the scheduler object (ReduceLROnPlateau in this project) and 'monitor' which is set to 'val_acc'.
    """
    # Instantiate optimizer and learning rate scheduler
    optimizer = optim.AdamW(
      params=params_group(self.model, self.hparams.weight_decay_rate_stage1),
      lr=self.hparams.learning_rate_stage1
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer=optimizer,
      mode='max',
      factor=0.1,
      patience=3,
      min_lr=1e-7
    )

    optimization_dict = {'optimizer': optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'monitor': 'val_acc'}}
    return optimization_dict



def instantiate_and_train_dataset_and_model(data_root: str, train_batch_size: int, val_batch_size: int, learning_rate_stage1: float,
                                              learning_rate_stage2: float, num_epochs: int,
                                              train_transform: torchvision.transforms.Compose,
                                              val_transform: torchvision.transforms.Compose, devices: int | str = 'auto',
                                              num_workers: int = 0, persistent_workers: bool = False, pin_memory: bool = False,
                                              prefetch_factor: int = 2, precision: str = '32-true', weight_decay_rate_stage1: float = 0.0,
                                              weight_decay_rate_stage2: float = 0.0, logger: bool = False,
                                              enable_model_summary: bool = False, enable_progress_bar: bool = True,
                                              enable_checkpointing: bool = True, accumulate_grad_batches: int = 1,
                                              train_dir: str = 'train', val_dir: str = 'test', num_classes: int = 2,
                                              weight_path: str = None, fast_dev_run: bool = False,
                                              save_weights_path: str | None = None) -> tuple[LightningDataModule, LightningModule, Trainer]:
    """
    This function instantiates data module, model and the trainer object with given arguments, train the model and return dataset, model
    and trainer object.

    Parameters
    ----------
    data_root : str
      The path to the dataset.
    train_batch_size : int
      Batch size of the train dataset.
    val_batch_size : int
      Batch size of the validation dataset.
    learning_rate_stage1 : float
      Learning rate for stage 1 of fine-tuning the model.
    learning_rate_stage2 : float
      Learning rate for stage 2 of fine-tuning the model.
    num_epochs : int
      Number of epochs to train the model.
    train_transform : torchvision.transforms.Compose
      Transforms used for the train dataset.
    val_transform : torchvision.transforms.Compose
      Transforms used for the validation dataset.
    devices : int | str, optional
      Number of GPUs available (default: 'auto').
    num_workers : int, optional
      Number of subprocesses used for parallel image loading (default: 0).
    persistent_workers : bool, optional
      If True, the data loader will not shut down the worker processes after a dataset has been consumed once (default: False).
    pin_memory : bool, optional
      If True, moves data to pinned (page-locked) memory for faster GPU transfer (default: False).
    prefetch_factor : int, optional
      Number of batches preloaded by each worker (default: 2).
    precision : str, optional
      Precision of matrix multiplication in the forward pass (default: '32-true').
    weight_decay_rate_stage1 : float, optional
      Weight decay rate of stage 1 of fine-tuning the model (default: 0.0).
    weight_decay_rate_stage2 : float, optional
      Weight decay rate of stage 2 of fine-tuning the model (default: 0.0).
    logger : bool, optional
      Whether to save logging of the trainer (default: False).
    enable_model_summary : bool, optional
      whether to print the summary of the model by trainer (default: False).
    enable_checkpointing : bool, optional
      Saves a checkpoint in the current working directory, with the state of the last training epoch (default: True).
    enable_progress_bar : bool, optional
      Whether to enable progress bar for training (default: True).
    accumulate_grad_batches : int, optional
      Number of batches to process before stepping the optimizer (default: 1).
    train_dir : str, optional
      Name of the train split of the dataset directory (default: 'train').
    val_dir : str, optional
      Name of the validation split of the dataset directory (default: test).
    num_classes : int, optional
      Number of classes (default: 2).
    weight_path : str, optional
      The path of the ResNet-50 pretrained weights on local storage (default: None).
    fast_dev_run : bool, optional
      If True, trainer qucikly runs one step of training to make sure everything works (default: False).
    save_weights_path : str | None, optional
      The path to save the weights of the trained model to (default: None).

    Returns
    -------
    dataset : LightningDataModule
        Instantiated Lightning DataModule used for training.
    model : LightningModule
        Trained LightningModule object.
    trainer : Trainer
        PyTorch Lightning Trainer object used for training.
    """
    
    # Instantiate the dataset using the data module class
    dataset = AIvsRealDataset(data_root=data_root, train_dir=train_dir, val_dir=val_dir, train_batch_size=train_batch_size,
                                val_batch_size=val_batch_size, num_workers=num_workers, persistent_workers=persistent_workers,
                                pin_memory=pin_memory, prefetch_factor=prefetch_factor, train_transform=train_transform,
                                val_transform=val_transform)

    # Load the model's weights from storage if they already exist.
    if save_weights_path is not None:
      if '.' in save_weights_path:
        if len(save_weights_path.split('.')) > 2:
          raise ValueError('When loading model weights from storage, only use . (dot) before extension.')
        else:
          save_weights_full_path = save_weights_path.split('.')[0]
          full_path = '.'.join([save_weights_full_path, 'pth'])
          if os.path.exists(full_path):
            print(f'Trained weights exist in path {full_path}. Loading from storage...')
            model = AIvsRealModel(learning_rate_stage1=learning_rate_stage1, weight_decay_rate_stage1=weight_decay_rate_stage1,
                                  weight_decay_rate_stage2=weight_decay_rate_stage2, learning_rate_stage2=learning_rate_stage2,
                                  num_epochs=num_epochs, num_classes=num_classes, weight_path=weight_path)
            
            model.load_state_dict(torch.load(full_path, map_location='cpu'))
            print('Loading finished.')
            return dataset, model, None
      else:
        full_path = '.'.join([save_weights_path, 'pth'])
        if os.path.exists(full_path):
          print(f'Trained weights exist in path {full_path}. Loading from storage...')
          model = AIvsRealModel(learning_rate_stage1=learning_rate_stage1, weight_decay_rate_stage1=weight_decay_rate_stage1,
                                weight_decay_rate_stage2=weight_decay_rate_stage2, learning_rate_stage2=learning_rate_stage2,
                                num_epochs=num_epochs, num_classes=num_classes, weight_path=weight_path)
          
          model.load_state_dict(torch.load(full_path, map_location='cpu'))
          print('Loading finished.')
          return dataset, model, None
    
    # Instantiate the model using the module class
    model = AIvsRealModel(learning_rate_stage1=learning_rate_stage1, weight_decay_rate_stage1=weight_decay_rate_stage1,
                            weight_decay_rate_stage2=weight_decay_rate_stage2, learning_rate_stage2=learning_rate_stage2,
                            num_epochs=num_epochs, num_classes=num_classes, weight_path=weight_path)
    
    # Instantiate the trainer.
    trainer = Trainer(
        accelerator='auto',
        devices=devices,
        precision=precision,
        logger=logger,
        fast_dev_run=fast_dev_run,
        max_epochs=num_epochs,
        enable_model_summary=enable_model_summary,
        enable_progress_bar=enable_progress_bar,
        accumulate_grad_batches=accumulate_grad_batches,
        enable_checkpointing=enable_checkpointing
    )

    # Train the model
    print('Training Started...')
    trainer.fit(model, dataset)
    print('Training Finished!')

    # Save trained model weights if needed. This part makes main directory and subdirectory on the fly, if provided, but the final
    # weight extension is always going to be pth.
    if save_weights_path is not None:
      save_weights_path = save_weights_path.lower().replace(' ', '_')
      if '/' in save_weights_path:
        save_weights_path_splits = save_weights_path.split('/')
        directories = '/'.join([path_split for path_split in save_weights_path_splits[:-1]])
        os.makedirs(directories, exist_ok=True)
        if '.' in save_weights_path_splits[-1]:
          if len(save_weights_path_splits[-1].split('.')) > 2:
            raise ValueError(f'When saving model weights to a file, only use . (dot) before extension.')
          save_weights_name = save_weights_path_splits[-1].split('.')[0]
          save_weights_name_ext = '.'.join([save_weights_name, 'pth'])
          torch.save(model.state_dict(), os.path.join(directories, save_weights_name_ext))
        else:
          save_weights_name_ext = '.'.join([save_weights_path_splits[-1], 'pth'])
          torch.save(model.state_dict(), os.path.join(directories, save_weights_name_ext))
      else:
        if '.' in save_weights_path:
          if len(save_weights_path.split('.')) > 2:
            raise ValueError(f'When saving model weights to a file, only use . (dot) before extension.')
          save_weights_name = save_weights_path.split('.')[0]
          save_weights_name_ext = '.'.join([save_weights_name, 'pth'])
          torch.save(model.state_dict(), save_weights_name_ext)
        else:
          save_weights_name_ext = '.'.join([save_weights_path, 'pth'])
          torch.save(model.state_dict(), save_weights_name_ext)

    return dataset, model, trainer



def hyperparameters_research(data_root: str, learning_rate_stage1_candidates: list, weight_decay_rate_stage1_candidates: list,
                              train_dir: str = 'train', val_dir: str = 'test', num_classes: int = 2,
                              precision: str = '32-true', logger: bool = False, enable_model_summary: bool = False,
                              enable_progress_bar: bool = True, enable_checkpointing: bool = False,
                              weight_path: str = 'resnet50_weights.pth', train_transform: transforms.Compose | None = None,
                              val_transform: transforms.Compose | None = None) -> dict:
  """
  Returns a fine combination of hyperparameters to use.

  This function reads the optimal data loader pipeline settings from a csv file, use it to instantiate the dataset and data loaders,
  performs a sequential (greedy) hyperparameter search, where each group of hyperparameters is optimized while keeping previously selected
  best values fixed, writes the results to a csv file and plots test results. Finally, it returns the final results as python dictionary.

  Parameters
  ----------
  data_root : str
    The path to the dataset.
  learning_rate_stage1_candidates : list
    Stage 1's learning rate candidate values to test.
  weight_decay_rate_stage1_candidates : list
    Stage 1's weight decay candidate values to test.
  train_dir : str, optional
    Name of the train split of the dataset directory (default: 'train').
  val_dir : str, optional
    Name of the validation split of the dataset directory (default: test).
  num_classes : int, optional
    Number of classes (default: 2).
  precision : str, optional
    Precision of matrix multiplication in the forward pass (default: '32-true').
  logger : bool, optional
    Whether to save logging of the trainer (default: False).
  enable_model_summary : bool, optional
    whether to print the summary of the model by trainer (default: False).
  enable_progress_bar : bool, optional
    Whether to enable progress bar for training (default: True).
  enable_checkpointing : bool, optional
    Saves a checkpoint in the current working directory, with the state of the last training epoch (default: True).
  weight_path : str, optional
    The path of the ResNet-50 pretrained weights on local storage (default: None).
  train_transform : torchvision.transforms.Compose | None, optional
    Transforms used for the train dataset (default: None).
  val_transform : torchvision.transforms.Compose | None, optional
    Transforms used for the validation dataset (default: None).

  Returns
  -------
  results : dict
    A python dictionary containing optimal values found for hyperparameters.
  """
  # Load the csv file containing optimal hyperparameters if it already exists in storage.
  if os.path.exists('optimal_hyperparameters.csv'):
      print('A csv file already exists containing the optimal hyperparameters.')
      print("Please use 'load_optimal_hyperparameters()' function to load the csv data into a python dictionary and proceed.")
      return
  
  # Read the optimal dataloader pipeline settings from csv file.
  dataloader_pipeline_settings = load_dataloader_optimal_settings()

  # Make empty dictionaries to store test results.
  learning_rate_stage1_results = {}
  weight_decay_rate_stage1_results = {}

  # Default values used in tests.
  weight_decay_rate_stage1_default = 1e-3
  num_epochs_default = 5
  accumulate_grad_batches_default = 2

  # Stage 1 learning rate test.
  for learning_rate_stage1 in learning_rate_stage1_candidates:
      # Instantiate the dataset and model with appropriate parameters and train the model.
      learning_rate_stage2 = learning_rate_stage1 * 0.1
      _, _, trainer = instantiate_and_train_dataset_and_model(
          data_root=data_root,
          train_batch_size=dataloader_pipeline_settings['train_batch_size'],
          val_batch_size=dataloader_pipeline_settings['train_batch_size'], # Val batch size is intentionally the same as train batch size!
          learning_rate_stage1=learning_rate_stage1,
          learning_rate_stage2=learning_rate_stage2,
          num_epochs=num_epochs_default,
          train_transform=train_transform,
          val_transform=val_transform,
          devices='auto',
          num_workers=dataloader_pipeline_settings['num_workers'],
          persistent_workers=dataloader_pipeline_settings['persistent_workers'],
          pin_memory=dataloader_pipeline_settings['pin_memory'],
          prefetch_factor=2,
          precision=precision,
          weight_decay_rate_stage1=weight_decay_rate_stage1_default,
          weight_decay_rate_stage2=weight_decay_rate_stage1_default, # In this project, weight_decay_rate_stage2=weight_decay_rate_stage1
          logger=logger,
          enable_model_summary=enable_model_summary,
          enable_progress_bar=enable_progress_bar,
          enable_checkpointing=enable_checkpointing,
          accumulate_grad_batches=accumulate_grad_batches_default,
          train_dir=train_dir,
          val_dir=val_dir,
          num_classes=num_classes,
          weight_path=weight_path,
          fast_dev_run=False,
      )

      # Store validation accuracy as the result of the current test.
      learning_rate_stage1_results[learning_rate_stage1] = trainer.callback_metrics['val_acc']
      print(f"Stage 1's Learning Rate = {learning_rate_stage1:.7f} results in Validation Accuracy {trainer.callback_metrics['val_acc']:.3%}")
      
      # Wait 5 seconds for user to see the previous log.
      time.sleep(5)
      
      # Empty GPU cache and garbage collect for efficiency and test isolation purposes.
      del trainer
      torch.cuda.empty_cache()
      gc.collect()

      # Clear all cell outputs.
      clear_output()

  # Find optimal index.
  max_idx = np.array(list(learning_rate_stage1_results.values())).argmax()

  # Choose optimal setting.
  best_learning_rate_stage1 = np.array(list(learning_rate_stage1_results.keys()))[max_idx]
  best_learning_rate_stage2 = best_learning_rate_stage1 * 0.1

  print(
    f"Best Stage 1's Learning Rate = {best_learning_rate_stage1:.7f} and "
    f"Best Stage 2's Learning Rate = {best_learning_rate_stage2:.7f} "
    f"results in Validation Accuracy {learning_rate_stage1_results[best_learning_rate_stage1]:.3%}"
  )
  time.sleep(5)
  clear_output()

  for weight_decay_rate_stage1 in weight_decay_rate_stage1_candidates:
      _, _, trainer = instantiate_and_train_dataset_and_model(
          data_root=data_root,
          train_batch_size=dataloader_pipeline_settings['train_batch_size'],
          val_batch_size=dataloader_pipeline_settings['train_batch_size'],
          learning_rate_stage1=best_learning_rate_stage1,
          learning_rate_stage2=best_learning_rate_stage2,
          num_epochs=num_epochs_default,
          train_transform=train_transform,
          val_transform=val_transform,
          devices='auto',
          num_workers=dataloader_pipeline_settings['num_workers'],
          persistent_workers=dataloader_pipeline_settings['persistent_workers'],
          pin_memory=dataloader_pipeline_settings['pin_memory'],
          prefetch_factor=2,
          precision=precision,
          weight_decay_rate_stage1=weight_decay_rate_stage1,
          weight_decay_rate_stage2=weight_decay_rate_stage1,
          logger=logger,
          enable_model_summary=enable_model_summary,
          enable_progress_bar=enable_progress_bar,
          enable_checkpointing=enable_checkpointing,
          accumulate_grad_batches=accumulate_grad_batches_default,
          train_dir=train_dir,
          val_dir=val_dir,
          num_classes=num_classes,
          weight_path=weight_path,
          fast_dev_run=False,
      )

      weight_decay_rate_stage1_results[weight_decay_rate_stage1] = trainer.callback_metrics['val_acc']
      print(
        f"Stages 1 and 2 Weight Decay = {weight_decay_rate_stage1:.7f} results in Validation "
        f"Accuracy {trainer.callback_metrics['val_acc']:.3%}"
      )
      time.sleep(5)
      del trainer
      torch.cuda.empty_cache()
      gc.collect()
      clear_output()

  max_idx = np.array(list(weight_decay_rate_stage1_results.values())).argmax()
  best_weight_decay_rate_stage1 = np.array(list(weight_decay_rate_stage1_results.keys()))[max_idx]
  best_weight_decay_rate_stage2 = best_weight_decay_rate_stage1

  print(
    f"Best Stage 1's Weight Decay = {best_weight_decay_rate_stage1:.7f} "
    f"and Best Stage 2's Weight Decay = {best_weight_decay_rate_stage2:.7f} "
    f"results in Validation Accuracy {weight_decay_rate_stage1_results[best_weight_decay_rate_stage1]:.3%}"
  )
  time.sleep(5)
  clear_output()

  results = {
    'learning_rate_stage1': best_learning_rate_stage1,
    'learning_rate_stage2': best_learning_rate_stage2,
    'weight_decay_rate_stage1': best_weight_decay_rate_stage1,
    'weight_decay_rate_stage2': best_weight_decay_rate_stage2,
  }

  # Write the final optimal settings into a csv file.
  with open('optimal_hyperparameters.csv', mode='w', newline='') as f:
     writer = csv.writer(f)
     writer.writerow(['Name', 'Value'])

     for k, v in results.items():
        writer.writerow([k, v])

  # Plot the test results.
  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 5), layout='constrained')

  axes[0].semilogx(list(learning_rate_stage1_results.keys()), list(learning_rate_stage1_results.values()), marker='o', markersize=6,
                          linestyle='--', c='blue')
  axes[0].axis('on')
  axes[0].set_xlabel("Stage 1's Learning Rate")
  axes[0].set_ylabel('Validation Accuracy')


  axes[1].semilogx(list(weight_decay_rate_stage1_results.keys()), list(weight_decay_rate_stage1_results.values()), marker='o',
                    markersize=6, linestyle='--', c='blue')
  axes[1].axis('on')
  axes[1].set_xlabel("Stage 1's Weight Decay Rate")
  axes[1].set_ylabel('Validation Accuracy')

  fig.suptitle('Hyperparameters Tuning')
  plt.show()
  plt.close('all')

  return results



def load_optimal_hyperparameters() -> dict:
  """
  Loads the optimal hyperparameters from the csv file.

  Returns
  -------
  optimal_hparams
    A python dictionary containing optimal hyperparameters.
  """
  optimal_hparams = {}
  with open('optimal_hyperparameters.csv', mode='r') as f:
    reader = csv.reader(f)
    next(reader)

    for row in reader:
      val = float(row[1])
      optimal_hparams[row[0]] = val

  return optimal_hparams