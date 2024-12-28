from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
import numpy as np

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
    random.seed(seed)  # Python 随机模块
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # 如果使用多卡训练
    torch.backends.cudnn.deterministic = True  # 确保 CUDNN 算法的确定性
    torch.backends.cudnn.benchmark = False  # 禁用 CUDNN 的优化算法
set_seed(173)

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate_rf(train_loader, test_loader, n_estimators=100, max_depth=None, random_state=42):
    """
    使用随机森林训练并评估模型，返回训练集正确率、测试集正确率和测试集每类指标。

    参数:
        train_loader: 训练集的 DataLoader。
        test_loader: 测试集的 DataLoader。
        n_estimators: 随机森林中决策树的数量 (默认=100)。
        max_depth: 决策树的最大深度 (默认=None，表示不限制)。
        random_state: 随机种子 (默认=42)。

    返回:
        train_accuracy: 训练集正确率。
        test_accuracy: 测试集正确率。
        per_class_metrics: 每类的精确率、召回率和F1分数。
    """
    # 准备训练数据
    X_train, y_train = [], []
    for X_batch, y_batch in train_loader:
        X_train.append(X_batch.view(X_batch.size(0), -1).numpy())  # 展平特征
        y_train.append(y_batch.numpy())
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0).astype(int)  # 确保为整数

    # 准备测试数据
    X_test, y_test = [], []
    for X_batch, y_batch in test_loader:
        X_test.append(X_batch.view(X_batch.size(0), -1).numpy())  # 展平特征
        y_test.append(y_batch.numpy())
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0).astype(int)  # 确保为整数

    # 初始化随机森林模型
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    # 训练模型
    rf_model.fit(X_train, y_train)

    # 预测训练集和测试集
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)

    # 计算训练集正确率
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # 计算测试集总体正确率
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # 计算分类报告（包含每类的Precision, Recall, F1）
    test_class_report = classification_report(
        y_test, y_test_pred, output_dict=True, zero_division=0
    )

    # 提取每类的指标
    per_class_metrics = {
        class_label: {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1-score": metrics["f1-score"],
        }
        for class_label, metrics in test_class_report.items()
        if class_label not in ["accuracy", "macro avg", "weighted avg"]  # 排除总体指标
    }

    return train_accuracy, test_accuracy, per_class_metrics

# 调用函数并打印结果
results = train_and_evaluate_rf(train_loader, test_loader, n_estimators=150, max_depth=10)
print(f"训练集正确率: {results[0]:.2f}")
print(f"测试集正确率: {results[1]:.2f}")
print("测试集每类指标:")
for class_label, metrics in results[2].items():
    print(f"  类别 {class_label}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1-score']:.4f}")
