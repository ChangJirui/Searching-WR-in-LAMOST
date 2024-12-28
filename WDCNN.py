import random
from model_functions import *

with open(r"/home/changjirui/rework/dataset_3/WN_generated.pkl", 'rb') as f:
    WN = pickle.load(f)
with open(r"/home/changjirui/rework/dataset_3/WC_WO_generated.pkl", 'rb') as f:
    WC_WO = pickle.load(f)
with open(r"/home/changjirui/rework/dataset_3/noise.pkl", 'rb') as f:
    noise = pickle.load(f)

dataset_O = create_dataset("/home/changjirui/rework/dataset_2/O/")
dataset_B = create_dataset("/home/changjirui/rework/dataset_2/B/")
dataset_A = create_dataset("/home/changjirui/rework/dataset_2/A/")
dataset_F = create_dataset("/home/changjirui/rework/dataset_2/F/")
dataset_G = create_dataset("/home/changjirui/rework/dataset_2/G/")
dataset_K = create_dataset("/home/changjirui/rework/dataset_2/K/")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(48)

def create_dataloader_2(WN, WC_WO, dataset_O, dataset_B, dataset_A, dataset_F, dataset_G, dataset_K, noise):
    WN_labels = list(np.zeros(len(WN), dtype=int))
    WC_WO_labels = list(np.ones(len(WC_WO), dtype=int))
    OBA_labels = list(np.ones(len(dataset_O) + len(dataset_B) + len(dataset_A),dtype=int) * 2)
    FGK_labels = list(np.ones(len(dataset_F) + len(dataset_G) + len(dataset_K),dtype=int) * 3)
    I_labels = list(np.ones(len(noise), dtype=int) * 4)
    labels = WN_labels + WC_WO_labels + OBA_labels + FGK_labels + I_labels

    processed_spectrum = WN + WC_WO + dataset_O + dataset_B + dataset_A + dataset_F + dataset_G + dataset_K + noise
    spec_data = torch.tensor(processed_spectrum, dtype=torch.float32)
    spec_labels = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(spec_data, spec_labels)
    dataset_size = len(dataset)
    train_size = int(0.6 * dataset_size)
    val_size = int(0.2 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = create_dataloader_2(WN, WC_WO, dataset_O, dataset_B, dataset_A, dataset_F, dataset_G, dataset_K,  noise)

class WDCNN(nn.Module):
    def __init__(self, output):
        super(WDCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=16, stride=1, padding=32)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=16)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=2, stride=1, padding=8)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(5824, 128)
        self.fc2 = nn.Linear(128, output)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

run_2(train_loader, val_loader, WDCNN(5), 0.00001, 100)