import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    # 读取训练集（补充 sep="," 和 skipinitialspace=True，避免逗号后空格导致列识别错误）
    train_df = pd.read_csv(
        os.path.join(data_dir, "train.txt"), 
        header=None, 
        names=["guid", "tag"],
        sep=",",  # 指定分隔符为逗号
        skipinitialspace=True  # 忽略逗号后的空格（适配 train.txt 格式）
    )
    # 划分训练集/验证集（8:2）—— 移除 stratify 参数，解决样本数不足问题
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    # 读取测试集（同样补充分隔符参数，避免格式错误）
    test_df = pd.read_csv(
        os.path.join(data_dir, "test_without_label.txt"), 
        header=None, 
        names=["guid", "tag"],
        sep=",",
        skipinitialspace=True
    )
    
    # 标签映射（str→int）
    tag2id = {"positive": 0, "neutral": 1, "negative": 2}
    train_df["label"] = train_df["tag"].map(tag2id)
    val_df["label"] = val_df["tag"].map(tag2id)
    
    return train_df, val_df, test_df, tag2id

# 示例调用
data_dir = "../data"  # 根据实际路径调整
train_df, val_df, test_df, tag2id = load_data(data_dir)