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

set_seed(537)

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



def validate_2(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).long()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    epoch_accuracy = accuracy_score(all_labels, all_preds)

    class_report = classification_report(
        all_labels, all_preds, output_dict=True, zero_division=0
    )

    per_class_metrics = {
        class_label: {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1-score": metrics["f1-score"],
        }
        for class_label, metrics in class_report.items()
        if class_label.isdigit()
    }

    return epoch_loss, epoch_accuracy, per_class_metrics

def train_and_validate_2(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    val_per_class_metrics = {str(i): {"precision": [], "recall": [], "f1-score": []} for i in range(5)}  # 假设5类

    for epoch in range(num_epochs):

        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)


        val_loss, val_accuracy, per_class_metrics = validate_2(model, val_loader, criterion, device)


        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)


        for class_label, metrics in per_class_metrics.items():
            for metric_name, value in metrics.items():
                val_per_class_metrics[class_label][metric_name].append(value)


        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        print('Per-Class Metrics:')
        for class_label, metrics in per_class_metrics.items():
            print(
                f'  Class {class_label} - Precision: {metrics["precision"]:.4f}, Recall: {metrics["recall"]:.4f}, F1-Score: {metrics["f1-score"]:.4f}'
            )

    print("Training and validation complete.")


    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(16, 12))


    plt.subplot(3, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


    plt.subplot(3, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.title('Overall Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()


    for i, metric_name in enumerate(["precision", "recall", "f1-score"], start=3):
        plt.subplot(3, 2, i)
        for class_label, metrics in val_per_class_metrics.items():
            plt.plot(epochs, metrics[metric_name], label=f'Class {class_label}')
        plt.title(metric_name.capitalize())
        plt.xlabel('Epoch')
        plt.ylabel(metric_name.capitalize())
        plt.legend()

    plt.tight_layout()
    plt.show()


    return train_losses, val_losses, val_accuracies, val_per_class_metrics

def run_2(train_loader, val_loader, model, learning_rate, num_epochs):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    train_and_validate_2(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

class CNN_LSTM_6(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CNN_LSTM_6, self).__init__()


        self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=64, stride=1, padding=32)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=32, stride=1, padding=16)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=24, stride=1, padding=8)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)


        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)


        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # CNN 部分
        x = x.unsqueeze(1)  # 加入一个通道维度
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.pool(x)

        #
        x = x.view(x.size(0), x.size(2), -1)  #

        #
        lstm_out, (h_n, c_n) = self.lstm(x)  #
        lstm_out = lstm_out[:, -1, :]  #

        #
        x = nn.functional.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

run_2(train_loader, val_loader, CNN_LSTM_6(64, 256, 5), 0.00001, 2500)