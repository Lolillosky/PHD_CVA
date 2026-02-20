import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import random_split

class DenseModel(nn.Module):
    def __init__(self, num_inputs, num_hidden_layers, num_neurons_per_hidden_layer, dtype=torch.float64):
        super().__init__()

        layers = []
        in_features = num_inputs

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_features, num_neurons_per_hidden_layer, dtype=dtype))
            layers.append(nn.Softplus())
            in_features = num_neurons_per_hidden_layer

        layers.append(nn.Linear(in_features, 1, dtype=dtype))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

class DiffDeepLearning(nn.Module):
    def __init__(self, model):
        super(DiffDeepLearning, self).__init__()
        
        self.model = model

    def forward(self, x):

        y = self.model(x)
        grad = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

        return torch.cat([y, grad], dim=1)
    

class DiffDeepLearningDataset(Dataset):
    def __init__(self, x_file, y_file):
        # Class constructor:
        # Loads the dataset from files:

        X = np.load(x_file)
        y = np.load(y_file)

        self.X_mu = np.mean(X, axis = 0)
        self.X_sigma = np.std(X, axis = 0) + 1e-8

        self.y_mu = np.mean(y[:,0])
        self.y_sigma = np.std(y[:,0]) + 1e-8

        y[:,1:] *= self.X_sigma / self.y_sigma  
        y[:,0] = (y[:,0] - self.y_mu) / self.y_sigma

        self.dydX_scaled_L2_norm = np.mean(y[:,1:]*y[:,1:], axis = 0) + 1e-8

        self.X = torch.tensor((X - self.X_mu) / self.X_sigma, dtype=torch.float64) 
        self.y = torch.tensor(y, dtype=torch.float64)

        
        self.len = self.X.shape[0]

        

    def __getitem__(self, index):
        # Retrieves an item from the dataset:
        # In very big daytasets, we should not load all data at once.

        return self.X[index], self.y[index]
    
        
    def __len__(self):
        # Returns the length of the dataset:
        return self.len
    

class DiffLearningLoss(nn.Module):

    def __init__(self, alpha, dydX_scaled_L2_norm):

        super().__init__()

        self.alpha = alpha

        norm_t = torch.as_tensor(dydX_scaled_L2_norm, dtype=torch.float64) 
        self.register_buffer("dydX_scaled_L2_norm", norm_t)

    def forward(self, pred, target):

        return torch.nn.functional.mse_loss(pred[:,0],target[:,0]) + self.alpha * torch.mean(torch.square(pred[:,1:]-target[:,1:])/self.dydX_scaled_L2_norm)
    

class DiffLearningFullModel:

    def __init__(self, num_inputs, num_hidden_layers, num_neurons_per_hidden_layer, alpha, train_dataloader, val_dataloader = None):
        
        self.model = DiffDeepLearning(DenseModel(num_inputs, num_hidden_layers, num_neurons_per_hidden_layer, torch.float64))

        self.alpha = alpha

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.get_normalization_params_set_loss()

        

    def get_normalization_params_set_loss(self):

        self.X_mu = self.train_dataloader.dataset.X_mu
        self.X_sigma = self.train_dataloader.dataset.X_sigma

        self.y_mu = self.train_dataloader.dataset.y_mu
        self.y_sigma = self.train_dataloader.dataset.y_sigma 

        self.dydX_scaled_L2_norm = self.train_dataloader.dataset.dydX_scaled_L2_norm

        self.loss_fn = DiffLearningLoss(self.alpha, self.train_dataloader.dataset.dydX_scaled_L2_norm)


    def train(self, epochs, writer=None):

        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch, (X, y) in enumerate(self.train_dataloader):
                X = X.requires_grad_(True)
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.train_dataloader)

            self.model.eval()
            val_loss = 0.0
            with torch.set_grad_enabled(True):
                for X, y in self.val_dataloader:
                    X = X.requires_grad_(True)
                    pred = self.model(X)
                    loss = self.loss_fn(pred, y)
                    val_loss += loss.item()
            val_loss /= len(self.val_dataloader)

            if writer:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)

            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    def forward(self, X):

        X_scaled = ((X - self.X_mu) / self.X_sigma)

        X_scaled.requires_grad_(True)

        y_sens_scaled_predicted = self.model.forward(X_scaled).detach().numpy()

        y = self.y_mu +  y_sens_scaled_predicted[:,0]*self.y_sigma

        sens = y_sens_scaled_predicted[:,1:] * self.y_sigma / self.X_sigma  

        return {'y': y, 'sens': sens}

    