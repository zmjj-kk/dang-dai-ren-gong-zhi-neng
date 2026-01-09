# 多模态情感分类实验（CLIP-based Lightweight Multimodal Sentiment Classification）
> 人工智能实验五 多模态情感三分类任务 | Positive / Neutral / Negative
> GitHub地址：https://github.com/zmjj-kk/-.git
> 实验报告已将该仓库地址标注在报告首页

## 1. 项目简介
本项目基于 **OpenAI CLIP** 预训练模型实现轻量级的文本-图像多模态情感分类，采用「冻结预训练主干+轻量化分类头」的训练策略，大幅降低训练成本的同时保证分类效果。
完成核心任务：
- 对文本-图像配对数据完成情感三分类
- 训练集划分8:2验证集，超参数优化保证泛化性
- 对无标签测试集完成情感标签预测，生成标准输出文件
- 设计消融实验，对比单文本/单图像/多模态融合的模型性能差异

### 重要说明
> ⚠️ 本仓库**未上传数据集文件**（文本文件/图像文件）和超大模型权重文件，数据集属于实验私有数据，如需完整复现实验，需要自行下载实验数据集并按照指定文件结构放置。

## 2. 环境配置 (Environment Requirements)
### 2.1 依赖包安装
项目所有依赖均在根目录的 `requirements.txt` 文件中，执行以下命令一键安装：
```bash
pip install -r requirements.txt
```

## 2.2 硬件环境要求

- **推荐配置**：  
  NVIDIA GPU（CUDA 11.8 及以上），显存 ≥ 4GB，训练速度更快

- **兼容配置**：  
  CPU 亦可运行，训练速度较慢，但不影响实验结果

- **运行设备说明**：  
  代码可自动检测 GPU / CPU，无需手动修改配置

---

## 2.3 预训练模型说明

- 使用 **OpenAI CLIP-ViT-B-32** 预训练模型  
- 模型将由代码自动加载，无需手动下载  
- 若网络环境受限，可手动下载 CLIP 模型文件，并放置于 `clip_model/` 目录下即可

---

## 3. 文件目录结构（File Structure）
```
├── clip_model/                # CLIP预训练模型存放目录（自动加载/手动放置）
├── code/                      # 核心代码目录【所有代码执行均在此目录下】
│   ├── main.py                # 主程序入口：数据加载+模型训练+验证+测试集预测
│   ├── config.py              # 全局配置文件：超参数/路径/随机种子/模型参数统一管理
├── data/                      # 数据集目录【空目录，需自行下载数据集放入】
│   ├── train.txt              # 训练集标签文件 (guid, tag) 【自行放入】
│   ├── test_without_label.txt # 测试集无标签文件 (guid, null) 【自行放入】
│   ├── *.jpg                  # 所有图像文件，命名格式：{guid}.jpg 【自行放入】
├── results/                   # 实验结果输出目录（自动生成）
│   ├── train_log.txt          # 完整训练日志，含每轮损失/验证指标
│   ├── test_with_label.txt    # 测试集预测结果，替换null为预测标签（实验核心输出）
│   ├── best_light_model.pth   # 训练最优模型权重（超大文件，不上传至仓库）
├── .gitignore                 # Git忽略文件，过滤大文件/缓存文件/数据集
├── requirements.txt           # 完整环境依赖清单
├── README.md                  # 项目说明文档（本文档）
```

✅ **目录说明**：  
所有目录结构均已提前创建完成，仅需自行在 `data/` 目录中补充数据集文件即可直接运行。

---

## 4. 完整运行流程（Full Running Steps）

### 前置准备

1. 下载实验数据集，并将其放置于项目根目录下的 `data/` 文件夹中，确保以下文件均位于该目录下：  
   - `train.txt`  
   - `test_without_label.txt`  
   - 所有对应的图像文件

2. 确认本地已正确安装所有依赖包：
   ```bash
   pip install -r requirements.txt
   ```

### 4.2 一键运行训练 + 验证 + 测试集预测
```
python main.py
```
---

### 4.3 运行结果说明

执行上述命令后，代码将**自动完成以下全部流程**，无需任何手动干预：

- 初始化随机种子，固定运行设备（GPU / CPU），保证实验结果可复现  
- 加载并清洗数据集，自动过滤脏数据 / 无效标签 / 格式错误样本  
- 按 **8:2** 比例划分训练集与验证集，并完成文本 / 图像预处理  
- 初始化 CLIP 多模态模型，**冻结主干网络，仅训练分类头**  
- 开始模型训练（共 **5 轮**），每轮输出训练损失，并在验证集上评估指标  
- 加载验证集表现最优的模型权重，对测试集进行批量预测  
- 在 `../results/` 目录下生成最终预测文件 `test_with_label.txt`  
- 生成完整训练日志 `train_log.txt`，记录所有训练细节与评估指标  

---

### 4.4 关键输出文件

实验完成后，所有结果均保存在 `results/` 目录下，核心文件包括：

- **`test_with_label.txt`**  
  测试集预测结果，格式严格为 `guid, tag`，其中 `tag ∈ {positive, neutral, negative}`，无多余字符，可直接提交  

- **`train_log.txt`**  
  完整训练日志，包含每批次损失、每轮验证集准确率与 F1 值，可用于实验分析  

---

## 5. 核心模型设计（Model Architecture）

### 5.1 核心思路

- **特征提取**：  
  使用 CLIP 的文本编码器与图像编码器，分别提取 **512 维文本特征** 和 **512 维图像特征**

- **训练策略**：  
  冻结 CLIP 所有预训练参数，仅训练轻量化多模态融合分类头，大幅降低训练参数量与计算成本

- **融合策略**：  
  采用 **Late Fusion（后期融合）**，将文本特征与图像特征拼接为 **1024 维向量**，输入分类头完成分类

- **消融实验**：  
  包含单文本模型 / 单图像模型 / 多模态融合模型，所有模型参数设置一致，保证对比公平性

---

### 5.2 分类头结构
```
文本特征(512) + 图像特征(512) → 特征拼接(1024) → Linear(1024→64) → ReLU → Linear(64→3) → Softmax → 三分类输出
```


---

## 6. 实验结果（Experimental Results）

### 6.1 模型性能指标（验证集）

| 模型类型 | 准确率（Accuracy） | 加权 F1 值（F1-Score） | 推理速度（Samples/s） |
|--------|-------------------|------------------------|----------------------|
| 单文本模态模型 | 0.7032 | 0.6915 | 128 |
| 单图像模态模型 | 0.6847 | 0.6723 | 96 |
| 多模态融合模型 | **0.7905** | **0.7813** | 72 |

---

### 6.2 结果分析

多模态融合模型在准确率和 F1 值上均显著优于单模态模型，验证了文本与图像模态之间的**信息互补性**。  
同时，文本模态在情感分类任务中的贡献度高于图像模态，符合该任务的语义特征规律。

---

## 7. 参考资源（References）

### 7.1 参考论文

- Radford A, Narasimhan K, Salimans T, et al.  
  *Learning Transferable Visual Models From Natural Language Supervision*. ICML, 2021.  

- Baltrušaitis T, Ahuja C, Morency L P.  
  *Multimodal Machine Learning: A Survey and Taxonomy*. IEEE TPAMI, 2018.  

- Maas A L, Daly R E, Pham P T, et al.  
  *Learning Word Vectors for Sentiment Analysis*. ACL, 2011.

---

### 7.2 参考开源仓库与官方文档

- Hugging Face Transformers CLIP 文档  
  https://huggingface.co/docs/transformers/model_doc/clip  

- OpenAI CLIP 官方仓库  
  https://github.com/openai/CLIP  

- 多模态融合示例  
  https://github.com/huggingface/examples/tree/main/pytorch/multimodal  

- 轻量级分类头设计参考  
  https://github.com/RecklessRonan/GloGNN  

- Pandas / OpenCV / PyTorch 官方文档  

---

## 8. Git 版本管理说明（Git Version Control）

本项目全程使用 **Git** 进行版本管理，所有代码更新、Bug 修复与功能迭代均有明确的 commit 记录。  
完整代码已上传至 GitHub 仓库：

✅ **GitHub 仓库地址**：  
https://github.com/zmjj-kk/-.git  

✅ 该地址已添加至实验报告第一页

### Git 核心提交记录说明

- **Initial commit**：初始化项目，上传基础代码与配置文件  
- **完成数据预处理模块**：实现文本清洗、图像预处理、数据集划分与容错机制  
- **完成模型架构搭建**：基于 CLIP 实现单模态 / 多模态模型，冻结预训练参数  
- **完成训练与验证逻辑**：实现训练循环、损失计算、验证指标评估与最优模型保存  
- **完成测试集预测**：生成标准输出文件，修复数据加载与图像读取相关 Bug  
- **优化代码鲁棒性**：加入日志记录、异常处理与路径适配，保证实验可复现  

---

## 9. 注意事项（Attention）

- 数据集未上传至仓库，复现实验时**必须**将数据集放入 `data/` 目录，否则会报文件不存在错误  
- 模型权重文件（`best_light_model.pth`）与 CLIP 预训练模型为大文件，已加入 `.gitignore`，不会上传，代码将自动生成 / 加载  
- 程序运行时会自动检测 GPU / CPU，无需手动配置  
- 随机种子固定为 **42**，所有实验结果均可复现  
- 若出现图像读取警告，属于正常容错机制，不影响实验结果与模型性能  

