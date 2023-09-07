import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import transforms, utils
from sentence_transformers import SentenceTransformer
from network.py import SHVNET
from PIL import Image

#dataset class of pytorch for images using pytorch
class ImageDataset(Dataset):
    def __init__(self):
        self.df = pd.read_csv('conTra_data.csv')
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img1 = self.df.iloc[idx, 0]
        img2 = self.df.iloc[idx, 2]
        lang1 = self.df.iloc[idx, 1]
        lang2 = self.df.iloc[idx, 3]
        action = self.df.iloc[idx, 4]
        img1 = Image.open(img1)
        img2 = Image.open(img2)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, lang1, img2, lang2, action

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=8, min_delta=-5):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=4, min_lr=1e-7, factor=0.1
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

def criterion(loss_func,outputs,pictures):
  losses = 0
  for i, key in enumerate(outputs):
        losses += loss_func(outputs[key], torch.squeeze(pictures[key].to(device),1))
  return losses

def train(model, train_dataloader, train_dataset, optimizer, loss_func):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataset)/train_dataloader.batch_size))
    for i, data in prog_bar:
        counter += 1
        data, target = data['train_row'].to(device), data['labels']
        total += data.size(0)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(loss_func, outputs, target)
        train_running_loss += loss.item()
        for i,key in enumerate(target):
          _, preds = torch.max(outputs[key], 1)
          train_running_correct += (preds == target[key]).sum().item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss / counter
    train_accuracy = 100. * train_running_correct / total
    return train_loss#, train_accuracy

def validate(model, test_dataloader, val_dataset, loss_func):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(test_dataloader), total=int(len(val_dataset)/test_dataloader.batch_size))
    with torch.no_grad():
        for i, data in prog_bar:
            counter += 1
            data, target = data['train_row'].to(device), data['labels']
            total += data.size(0)
            outputs = model(data)
            loss = criterion(loss_func,outputs, target)
            
            val_running_loss += loss.item()
            # _, preds = torch.max(outputs.data, 1)
            # val_running_correct += (preds == target).sum().item()
        
        val_loss = val_running_loss / counter
        #val_accuracy = 100. * val_running_correct / total
        return val_loss#, val_accuracy

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    epochs = 100
    learning_rate = 1e-3
    patience = 5
    min_delta = -5
    min_lr = 1e-7
    factor = 0.1
    # Load data
    train_set = TrainDataset()
    val_set = ValDataset()
    # Create data loaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=4
    )
    # Create model
    model = Model().to(device)
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Create loss function
    loss_func = nn.CrossEntropyLoss()
    # Create learning rate scheduler
    lr_scheduler = LRScheduler(optimizer, patience, min_lr, factor)
    # Create early stopping
    early_stopping = EarlyStopping(patience, min_delta)
    # Train model
    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []
    start = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss = train(
            model, train_loader, train_set, optimizer, loss_func
        )
        val_epoch_loss = validate(
            model, val_loader, val_set, loss_func
        )
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)
        lr_scheduler(val_epoch_loss)
        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            break
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f'Val Loss: {val_epoch_loss:.4f}')
    end = time.time()
    print(f"Training time: {(end-start)/60:.3f} minutes")
    print('Saving loss and accuracy plots...')
    #accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_accuracy, color='green', label='train accuracy')
    plt.plot(val_accuracy, color='blue', label='validataion accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"../outputs/{acc_plot_name}.png")
    plt.show()
    #loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
        
    # serialize the model to disk
    print('Saving model...')
    torch.save(model, '/content/model3.pt')
    
    print('TRAINING COMPLETE')