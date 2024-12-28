import torch
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
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
    random.seed(seed)  #
    np.random.seed(seed)  #
    torch.manual_seed(seed)  #
    torch.cuda.manual_seed(seed)  #
    torch.cuda.manual_seed_all(seed)  #
    torch.backends.cudnn.deterministic = True  #
    torch.backends.cudnn.benchmark = False  #

# 设置随机数种子
set_seed(2)

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
# 数据预处理函数
def dataloader_to_numpy(dataloader):

    features, labels = [], []
    for batch in dataloader:
        x, y = batch
        features.append(x.numpy())
        labels.append(y.numpy())

    features = torch.cat([torch.tensor(f) for f in features]).numpy()
    labels = torch.cat([torch.tensor(l) for l in labels]).numpy()
    return features, labels


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score
import torch


from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate_svm(train_loader, test_loader):

    train_features = []
    train_labels = []
    for x, y in train_loader:
        train_features.append(x.view(x.size(0), -1).numpy())  #
        train_labels.append(y.numpy())
    train_features = np.vstack(train_features)  #
    train_labels = np.hstack(train_labels).astype(int)  #


    test_features = []
    test_labels = []
    for x, y in test_loader:
        test_features.append(x.view(x.size(0), -1).numpy())
        test_labels.append(y.numpy())
    test_features = np.vstack(test_features)  #
    test_labels = np.hstack(test_labels).astype(int)  #

    # 创建并训练 SVM 模型
    model = SVC(kernel='rbf', C=1.0, random_state=42)
    model.fit(train_features, train_labels)


    train_preds = model.predict(train_features)
    test_preds = model.predict(test_features)


    train_accuracy = accuracy_score(train_labels, train_preds)


    test_accuracy = accuracy_score(test_labels, test_preds)


    test_class_report = classification_report(
        test_labels, test_preds, output_dict=True, zero_division=0
    )

    per_class_metrics = {
        class_label: {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1-score": metrics["f1-score"],
        }
        for class_label, metrics in test_class_report.items()
        if class_label not in ["accuracy", "macro avg", "weighted avg"]
    }

    return train_accuracy, test_accuracy, per_class_metrics



train_accuracy, test_accuracy, per_class_metrics = train_and_evaluate_svm(train_loader, test_loader)

print(f"Train accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print("Performance indicators:")
for class_label, metrics in per_class_metrics.items():
    print(f"  Class {class_label}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1-score']:.4f}")
