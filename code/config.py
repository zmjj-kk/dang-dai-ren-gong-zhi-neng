import torch
from torchvision import transforms

# -------------------------- 基础配置（轻量化核心） --------------------------
# 数据路径（确保data与code同级）
DATA_DIR = "../data"
RESULTS_DIR = "../results"
# 简化文本长度（降低编码耗时）
MAX_SEQ_LEN = 77
# 训练轮次减半（5轮足够收敛）
EPOCHS = 5
# 批次大小（CPU设8，GPU设16，平衡速度与内存）
BATCH_SIZE = 16 if torch.cuda.is_available() else 8
# 学习率适配轻量训练（无需微调预训练层，用1e-3更快收敛）
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# -------------------------- 模型配置（冻结CLIP） --------------------------
CLIP_MODEL_NAME = "../clip_model"
NUM_CLASSES = 3
# 模型权重保存路径
SAVE_MODEL_PATH = "../results/best_light_model.pth"

# -------------------------- 图像预处理（简化增强，降低CPU负担） --------------------------
# 训练集：仅保留水平翻转（移除旋转，减少计算）
train_img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 验证/测试集：无任何增强
val_test_img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])