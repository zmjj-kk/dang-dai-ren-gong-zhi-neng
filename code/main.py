import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # 进度条日志
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import CLIPModel, CLIPProcessor
import pandas as pd
import os
import logging
from PIL import Image
import re
import numpy as np  # 新增：用于生成随机图像
import config  # 导入轻量化配置

# -------------------------- 1. 初始化日志系统（完整日志输出） --------------------------
def init_logger():
    logger = logging.getLogger("LightMultimodal")
    logger.setLevel(logging.INFO)
    # 日志格式：时间+模块+级别+内容
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    # 文件输出（保存到results目录）
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(config.RESULTS_DIR, "train_log.txt"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

# 初始化日志（全局可用）
logger = init_logger()

# -------------------------- 2. 数据预处理（简化逻辑，降低负担） --------------------------
# 修复nltk下载问题：禁用自动下载，本地加载/替代
try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))
except ImportError:
    # 无nltk时用简单分词替代
    def word_tokenize(text):
        return text.split()
    stop_words = set()  # 空停用词表

def clean_text(text):
    """简化文本清洗：仅保留核心步骤，减少计算"""
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # 移除表情
    text = re.sub(r"#\w+", " ", text)  # 移除标签
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [tok for tok in tokens if tok.isalpha() and tok not in stop_words]
    # 截断到CLIP支持的77个token（含特殊符号，实际取75个，留2个位置）
    return " ".join(tokens[:config.MAX_SEQ_LEN - 2])

def load_text(guid, data_dir):
    """读取文本，添加错误日志"""
    text_path = os.path.join(data_dir, f"{guid}.txt")
    try:
        with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
            return clean_text(f.read().strip())
    except Exception as e:
        logger.warning(f"读取文本{guid}.txt失败：{str(e)}，返回空文本")
        return ""

def load_image(guid, data_dir, processor):
    """读取图像，修复CLIPProcessor的size参数问题"""
    img_path = os.path.join(data_dir, f"{guid}.jpg")
    try:
        image = Image.open(img_path).convert("RGB")
        # 修复：使用CLIP默认的尺寸处理（移除自定义size参数）
        image_inputs = processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            do_center_crop=True
            # 移除错误的size参数，使用CLIP默认的224x224
        )
        return image_inputs["pixel_values"].squeeze(0)
    except Exception as e:
        logger.warning(f"读取图像{guid}.jpg失败：{str(e)}，返回随机图像")
        # 生成符合CLIP要求的随机图像张量（3通道，224x224）
        return torch.randn(3, 224, 224)

def load_data(data_dir):
    """加载数据，过滤脏数据，打印分布日志"""
    logger.info(f"开始加载数据，路径：{data_dir}")
    # 读取训练集（修复engine参数+跳过坏行）
    train_path = os.path.join(data_dir, "train.txt")
    train_df = pd.read_csv(
        train_path, 
        header=None, 
        names=["guid", "tag"], 
        sep=",", 
        skipinitialspace=True,
        engine='python',  # 新增：指定python引擎
        on_bad_lines=lambda x: None  # 跳过格式错误行
    )
    
    # 过滤脏数据
    valid_tags = ["positive", "neutral", "negative"]
    train_df = train_df[train_df["tag"].isin(valid_tags)]  # 有效标签
    train_df = train_df[train_df["guid"] != "guid"]  # 移除表头行
    train_df = train_df.dropna(subset=["guid"])  # 移除空guid
    train_df = train_df.reset_index(drop=True)  # 重置索引

    # 划分训练/验证集（8:2）
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=config.SEED
    )
    val_df = val_df.reset_index(drop=True)

    # 读取测试集并修复engine参数
    test_path = os.path.join(data_dir, "test_without_label.txt")
    test_df = pd.read_csv(
        test_path, 
        header=None, 
        names=["guid", "tag"], 
        sep=",", 
        skipinitialspace=True,
        engine='python',  # 新增：指定python引擎
        on_bad_lines=lambda x: None  # 跳过格式错误行
    )
    test_df = test_df[test_df["guid"] != "guid"]
    test_df = test_df.dropna(subset=["guid"])
    test_df = test_df.reset_index(drop=True)

    # 标签映射
    tag2id = {"positive": 0, "neutral": 1, "negative": 2}
    train_df["label"] = train_df["tag"].map(tag2id)
    val_df["label"] = val_df["tag"].map(tag2id)
    
    # 移除映射后空值
    train_df = train_df.dropna(subset=["label"])
    val_df = val_df.dropna(subset=["label"])

    # 打印日志
    logger.info(f"数据加载完成：训练集{len(train_df)}条，验证集{len(val_df)}条，测试集{len(test_df)}条")
    logger.info(f"训练集标签分布：{train_df['tag'].value_counts().to_dict()}")
    return train_df, val_df, test_df, tag2id

# 自定义数据集（修复NoneType问题）
class LightMultimodalDataset(Dataset):
    def __init__(self, df, data_dir, processor):
        self.df = df
        self.data_dir = data_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        guid = str(row["guid"]).strip()
        label = row["label"] if "label" in self.df.columns else -1

        # 核心修复：所有场景都返回有效label张量（无None）
        if pd.isna(label) or label not in [0, 1, 2]:
            label = 1  # 无效/测试集默认中性
        label = int(label)
        label_tensor = torch.tensor(label, dtype=torch.long)

        # 读取并处理文本
        text = load_text(guid, self.data_dir)
        text_inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=config.MAX_SEQ_LEN
        )
        text_input_ids = text_inputs["input_ids"].squeeze(0)
        text_attention_mask = text_inputs["attention_mask"].squeeze(0)

        # 读取并处理图像
        image = load_image(guid, self.data_dir, self.processor)

        return {
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "image": image,
            "label": label_tensor,
            "guid": guid
        }

# -------------------------- 3. 轻量模型定义（修复单模态参数问题） --------------------------
class LightMultimodalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 加载CLIP并冻结全部层
        self.clip = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip_hidden_dim = 512

        # 简化融合分类头
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.clip_hidden_dim * 2, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, config.NUM_CLASSES)
        )
        logger.info("轻量模型初始化完成：CLIP全部冻结，仅训练融合分类头")

    def forward(self, text_input_ids, text_attention_mask, image):
        text_features = self.clip.get_text_features(input_ids=text_input_ids, attention_mask=text_attention_mask)
        image_features = self.clip.get_image_features(pixel_values=image)
        fused = torch.cat([text_features, image_features], dim=1)
        return self.classifier(fused)

# 单模态模型（修复参数不匹配问题）
class TextOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
        for param in self.clip.parameters():
            param.requires_grad = False
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, config.NUM_CLASSES)
        )

    # 修复：添加image=None兼容多模态调用
    def forward(self, text_input_ids, text_attention_mask, image=None):
        text_features = self.clip.get_text_features(input_ids=text_input_ids, attention_mask=text_attention_mask)
        return self.classifier(text_features)

class ImageOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
        for param in self.clip.parameters():
            param.requires_grad = False
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, config.NUM_CLASSES)
        )

    # 核心修复：把有默认值的参数放在无默认值参数后面
    def forward(self, image, text_input_ids=None, text_attention_mask=None):
        image_features = self.clip.get_image_features(pixel_values=image)
        return self.classifier(image_features)

# -------------------------- 4. 训练/验证工具 --------------------------
def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    logger.info(f"已设置随机种子：{seed}，确保结果可复现")

def train_epoch(model, dataloader, optimizer, device, epoch):
    """训练一轮"""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1} Train")):
        text_ids = batch["text_input_ids"].to(device)
        text_mask = batch["text_attention_mask"].to(device)
        image = batch["image"].to(device)
        labels = batch["label"].to(device)

        # 前向+反向
        optimizer.zero_grad()
        logits = model(text_ids, text_mask, image)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 每10个batch打印损失
        if (batch_idx + 1) % 10 == 0:
            avg_batch_loss = total_loss / (batch_idx + 1)
            logger.info(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} | 平均损失：{avg_batch_loss:.4f}")

    # 计算指标
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    logger.info(f"Epoch {epoch+1} Train | 损失：{avg_loss:.4f} | 准确率：{acc:.4f} | F1：{f1:.4f}")
    return avg_loss, acc, f1

def val_epoch(model, dataloader, device, mode="Val"):
    """验证/测试"""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_guids = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{mode} Process"):
            text_ids = batch["text_input_ids"].to(device)
            text_mask = batch["text_attention_mask"].to(device)
            image = batch["image"].to(device)
            labels = batch["label"].to(device) if mode != "Test" else None

            # 前向计算
            logits = model(text_ids, text_mask, image)
            if mode != "Test":
                loss = F.cross_entropy(logits, labels)
                total_loss += loss.item()
                all_labels.extend(labels.cpu().numpy())

            # 统计预测
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_guids.extend(batch["guid"])

    # 返回结果
    if mode != "Test":
        avg_loss = total_loss / len(dataloader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        logger.info(f"{mode} Result | 损失：{avg_loss:.4f} | 准确率：{acc:.4f} | F1：{f1:.4f}")
        return avg_loss, acc, f1
    else:
        logger.info(f"{mode} Process Done | 共预测{len(all_guids)}条数据")
        return all_guids, all_preds

# -------------------------- 5. 消融实验（可选） --------------------------
def run_ablation(device, train_loader, val_loader):
    logger.info("\n=== 开始消融实验（单模态模型，3轮训练）===")
    ablation_results = []

    # 单文本模型
    text_model = TextOnlyModel().to(device)
    text_opt = torch.optim.AdamW(text_model.parameters(), lr=config.LEARNING_RATE)
    for epoch in range(3):
        text_model.train()
        for batch in tqdm(train_loader, desc=f"Text-Only Epoch {epoch+1}"):
            text_opt.zero_grad()
            logits = text_model(batch["text_input_ids"].to(device), batch["text_attention_mask"].to(device))
            loss = F.cross_entropy(logits, batch["label"].to(device))
            loss.backward()
            text_opt.step()
    text_loss, text_acc, text_f1 = val_epoch(text_model, val_loader, device, mode="Text-Only Val")
    ablation_results.append({"模型": "单文本", "准确率": text_acc, "F1": text_f1})

    # 单图像模型
    img_model = ImageOnlyModel().to(device)
    img_opt = torch.optim.AdamW(img_model.parameters(), lr=config.LEARNING_RATE)
    for epoch in range(3):
        img_model.train()
        for batch in tqdm(train_loader, desc=f"Image-Only Epoch {epoch+1}"):
            img_opt.zero_grad()
            logits = img_model(batch["image"].to(device))
            loss = F.cross_entropy(logits, batch["label"].to(device))
            loss.backward()
            img_opt.step()
    img_loss, img_acc, img_f1 = val_epoch(img_model, val_loader, device, mode="Image-Only Val")
    ablation_results.append({"模型": "单图像", "准确率": img_acc, "F1": img_f1})

    # 保存结果
    ablation_df = pd.DataFrame(ablation_results)
    ablation_path = os.path.join(config.RESULTS_DIR, "ablation_results.csv")
    ablation_df.to_csv(ablation_path, index=False)
    logger.info(f"消融实验完成，结果保存到：{ablation_path}")
    return ablation_df

# -------------------------- 6. 主流程 --------------------------
def main():
    # 初始化
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    logger.info(f"=== 轻量多模态情感分类训练启动 ===")
    logger.info(f"设备：{device} | 批次大小：{config.BATCH_SIZE} | 训练轮次：{config.EPOCHS}")

    # 1. 加载数据
    try:
        train_df, val_df, test_df, tag2id = load_data(config.DATA_DIR)
        id2tag = {v: k for k, v in tag2id.items()}
    except Exception as e:
        logger.error(f"数据加载失败，程序终止：{str(e)}")
        return

    # 2. 初始化处理器和数据集
    logger.info("初始化CLIP处理器（首次运行需下载模型，约500MB）")
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)

    # 创建数据集（强制单线程）
    num_workers = 0
    train_dataset = LightMultimodalDataset(train_df, config.DATA_DIR, processor)
    val_dataset = LightMultimodalDataset(val_df, config.DATA_DIR, processor)
    test_dataset = LightMultimodalDataset(test_df, config.DATA_DIR, processor)

    # 自定义collate_fn（兜底处理）
    def custom_collate_fn(batch):
        batch = [item for item in batch if item is not None]
        collated = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                values = [v if v is not None else torch.zeros_like(values[0]) for v in values]
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
        return collated

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    logger.info("数据集和DataLoader创建完成")

    # 3. 训练多模态模型
    model = LightMultimodalModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    best_val_f1 = 0.0
    train_history = []

    logger.info("\n=== 开始多模态模型训练 ===")
    for epoch in range(config.EPOCHS):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_acc, val_f1 = val_epoch(model, val_loader, device)

        # 保存最优模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), config.SAVE_MODEL_PATH)
            logger.info(f"✅ 保存最优模型（验证集F1：{best_val_f1:.4f}）到 {config.SAVE_MODEL_PATH}")

        # 记录历史
        train_history.append({
            "epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1
        })

    # 保存训练历史
    history_df = pd.DataFrame(train_history)
    history_path = os.path.join(config.RESULTS_DIR, "val_results.csv")
    history_df.to_csv(history_path, index=False)
    logger.info(f"\n训练完成！训练历史保存到：{history_path}")
    logger.info(f"多模态模型最优验证集结果：准确率{val_acc:.4f} | F1{best_val_f1:.4f}")

    # 4. 跳过消融实验（已注释）
    # run_ablation(device, train_loader, val_loader)

    # 5. 测试集预测
    logger.info("\n=== 开始测试集预测 ===")
    model.load_state_dict(torch.load(config.SAVE_MODEL_PATH))
    test_guids, test_preds = val_epoch(model, test_loader, device, mode="Test")
    # 转换标签
    test_pred_tags = [id2tag[p] for p in test_preds]
    # 保存测试结果
    test_result_df = pd.DataFrame({"guid": test_guids, "tag": test_pred_tags})
    test_result_path = os.path.join(config.RESULTS_DIR, "test_with_label.txt")
    test_result_df.to_csv(test_result_path, index=False, header=False, sep=",")
    logger.info(f"测试结果保存到：{test_result_path}")
    logger.info(f"测试集标签分布：{test_result_df['tag'].value_counts().to_dict()}")

    # 6. 输出文件清单
    logger.info("\n=== 所有流程完成，生成文件清单 ===")
    logger.info(f"1. 训练日志：{os.path.join(config.RESULTS_DIR, 'train_log.txt')}")
    logger.info(f"2. 验证集结果：{history_path}")
    logger.info(f"3. 测试集预测：{test_result_path}")
    logger.info(f"4. 最优模型权重：{config.SAVE_MODEL_PATH}")

if __name__ == "__main__":
    main()