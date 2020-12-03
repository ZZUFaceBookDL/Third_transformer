import datetime

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import seaborn as sns

from tst import Transformer
from tst.loss import OZELoss

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src.dataset import OzeDataset
from src.metrics import MSE
from sklearn.metrics import r2_score
from src.utils import compute_loss, fit, Logger, kfold
from src.utils import compute_loss
from src.visualization import map_plot_function, plot_values_distribution, plot_error_distribution, plot_errors_threshold, plot_visual_sample

# Training parameters
DATASET_PATH = 'datasets/dataset_57M.npz'
BATCH_SIZE = 8
NUM_WORKERS = 0
LR = 2e-4
EPOCHS = 30

# Model parameters
d_model = 64 # Lattent dim
q = 8 # Query size
v = 8 # Value size
h = 8 # Number of heads
N = 4 # Number of encoder and decoder to stack
attention_size = 12 # Attention window size
dropout = 0.2 # Dropout rate
pe = None # Positional encoding
chunk_mode = None

d_input = 1460 # From dataset
d_output = 32 # From dataset

# Config
sns.set()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

class UcrDataset_train(Dataset):
    def __init__(self,filepath):
        xy = np.loadtxt(filepath,delimiter='\t',dtype=np.float32)
        # 此时取出来的数据是一个(N,M)的元组，N是行，M是列，xy.shape[0]可以把N取出来，也就是一共有多少行，有多少条数据
        self.len = xy.shape[1]
        self.x_data = torch.from_numpy(xy[0:60,1:])
        self.y_data = torch.from_numpy(xy[0:60,0])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class UcrDataset_val(Dataset):
    def __init__(self,filepath):
        xy = np.loadtxt(filepath,delimiter='\t',dtype=np.float32)
        # 此时取出来的数据是一个(N,M)的元组，N是行，M是列，xy.shape[0]可以把N取出来，也就是一共有多少行，有多少条数据
        self.len = xy.shape[1]
        self.x_data = torch.from_numpy(xy[60:100,1:])
        self.y_data = torch.from_numpy(xy[60:100,0])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class UcrDataset_test(Dataset):
    def __init__(self,filepath):
        xy = np.loadtxt(filepath,delimiter='\t',dtype=np.float32)
        # 此时取出来的数据是一个(N,M)的元组，N是行，M是列，xy.shape[0]可以把N取出来，也就是一共有多少行，有多少条数据
        self.len = xy.shape[1]
        self.x_data = torch.from_numpy(xy[:,1:])
        self.y_data = torch.from_numpy(xy[:1,0])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset_train = UcrDataset_train('F:\Data\\UCRArchive_2018\ACSF1\ACSF1_TRAIN.tsv')
train_loader = DataLoader(dataset=dataset_train,batch_size=BATCH_SIZE,shuffle=False)

dataset_test = UcrDataset_test('F:\Data\\UCRArchive_2018\ACSF1\ACSF1_TEST.tsv')
test_loader = DataLoader(dataset=dataset_test,batch_size=BATCH_SIZE,shuffle=False)

dataset_val = UcrDataset_val('F:\Data\\UCRArchive_2018\ACSF1\ACSF1_TRAIN.tsv')
val_loader = DataLoader(dataset=dataset_val,batch_size=BATCH_SIZE,shuffle=False)
# Load dataset
# ozeDataset = OzeDataset(DATASET_PATH)

# Split between train, validation and test
# dataset_train, dataset_val, dataset_test = random_split(
#     ozeDataset, (6500, 500, 500))
#
# dataloader_train = DataLoader(dataset_train,
#                               batch_size=BATCH_SIZE,
#                               shuffle=True,
#                               num_workers=NUM_WORKERS,
#                               pin_memory=False
#                               )
#
# dataloader_val = DataLoader(dataset_val,
#                             batch_size=BATCH_SIZE,
#                             shuffle=True,
#                             num_workers=NUM_WORKERS
#                             )
#
# dataloader_test = DataLoader(dataset_test,
#                              batch_size=BATCH_SIZE,
#                              shuffle=False,
#                              num_workers=NUM_WORKERS
#                              )

# Load transformer with Adam optimizer and MSE loss function
net = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size,
                  dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)
optimizer = optim.Adam(net.parameters(), lr=LR)
loss_function = OZELoss(alpha=0.3)

metrics = {
    'training_loss': lambda y_true, y_pred: OZELoss(alpha=0.3, reduction='none')(y_true, y_pred).numpy(),
    'mse_tint_total': lambda y_true, y_pred: MSE(y_true, y_pred, idx_label=[-1], reduction='none'),
    'mse_cold_total': lambda y_true, y_pred: MSE(y_true, y_pred, idx_label=[0, 1, 2, 3, 4, 5, 6], reduction='none'),
    'mse_tint_occupation': lambda y_true, y_pred: MSE(y_true, y_pred, idx_label=[-1], reduction='none', occupation=occupation),
    'mse_cold_occupation': lambda y_true, y_pred: MSE(y_true, y_pred, idx_label=[0, 1, 2, 3, 4, 5, 6], reduction='none', occupation=occupation),
    'r2_tint': lambda y_true, y_pred: np.array([r2_score(y_true[:, i, -1], y_pred[:, i, -1]) for i in range(y_true.shape[1])]),
    'r2_cold': lambda y_true, y_pred: np.array([r2_score(y_true[:, i, 0:-1], y_pred[:, i, 0:-1]) for i in range(y_true.shape[1])])
}

# logger = Logger(f'logs/training.csv', model_name=net.name,
#                 params=[y for key in metrics.keys() for y in (key, key+'_std')])

# Fit model
with tqdm(total=EPOCHS) as pbar:
    loss = fit(net, optimizer, loss_function, train_loader,
               val_loader, epochs=EPOCHS, pbar=pbar, device=device)

# Switch to evaluation
_ = net.eval()

# Select target values in test split
y_true = ozeDataset._y[dataloader_test.dataset.indices]

# Compute predictions
predictions = torch.empty(len(dataloader_test.dataset), 168, 8)
idx_prediction = 0
with torch.no_grad():
    for x, y in tqdm(dataloader_test, total=len(dataloader_test)):
        netout = net(x.to(device)).cpu()
        predictions[idx_prediction:idx_prediction+x.shape[0]] = netout
        idx_prediction += x.shape[0]

# Compute occupation times
occupation = ozeDataset._x[dataloader_test.dataset.indices,
                           :, ozeDataset.labels['Z'].index('occupancy')]

results_metrics = {
    key: value for key, func in metrics.items() for key, value in {
        key: func(y_true, predictions).mean(),
        key+'_std': func(y_true, predictions).std()
    }.items()
}

# Log
logger.log(**results_metrics)

# Save model
torch.save(net.state_dict(),
           f'models/{net.name}_{datetime.datetime.now().strftime("%Y_%m_%d__%H%M%S")}.pth')
