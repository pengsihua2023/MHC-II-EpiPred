import ast
import collections
import matplotlib.pyplot as plt
import numpy as np


# 文件名
log_file = 'II-650M-4H100.34716594-2.txt'

# 存放数据的列表
training_logs = []
eval_logs = []

# 逐行读取文件，只处理以 '{' 开头的行（即字典格式的日志）
with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            # 安全地将字符串转换为字典
            log_entry = ast.literal_eval(line)
        except Exception as e:
            # 无法解析的行跳过
            continue
        # 如果含有 'eval_loss' 则视为评估日志，否则视为训练日志（含 'loss'）
        if 'eval_loss' in log_entry:
            eval_logs.append(log_entry)
        elif 'loss' in log_entry:
            training_logs.append(log_entry)

# -------------------------
# 处理训练日志：按整数 epoch 分组，计算平均训练 loss
training_loss_by_epoch = collections.defaultdict(list)
for entry in training_logs:
    # 将 epoch 取整（例如 1.01、1.03 等都归为 epoch 1）
    epoch_group = int(entry['epoch'])
    training_loss_by_epoch[epoch_group].append(entry['loss'])

# 排序并计算每个 epoch 的平均 loss
train_epochs = sorted(training_loss_by_epoch.keys())
train_loss_avg = [np.mean(training_loss_by_epoch[ep]) for ep in train_epochs]

# -------------------------
# 处理评估日志：直接取日志中记录的 eval_loss 与 eval_accuracy
eval_epochs = []
eval_loss = []
eval_accuracy = []
for entry in eval_logs:
    # 这里假设评估日志中 'epoch' 字段是整数（如 1.0, 2.0, ...）
    eval_epochs.append(entry['epoch'])
    eval_loss.append(entry['eval_loss'])
    eval_accuracy.append(entry['eval_accuracy'])

# -------------------------
# 绘图

# 绘制 Training Loss vs. Evaluation Loss 曲线
plt.figure(figsize=(10, 6))
plt.plot(train_epochs, train_loss_avg, marker='o', color='blue', label='Training Loss (avg per epoch)')
plt.plot(eval_epochs, eval_loss, marker='o', color='red', label='Evaluation Loss')
plt.xlabel('Epoch', fontsize=16)           # 设置 X 轴标签的字体大小
plt.ylabel('Loss', fontsize=16)            # 设置 Y 轴标签的字体大小
plt.title('Training Loss vs. Evaluation Loss', fontsize=16)  # 设置标题的字体大小
plt.legend(fontsize=18)                    # 设置图例的字体大小
plt.grid(True)
plt.show()

# 绘制 Evaluation Accuracy 曲线
plt.figure(figsize=(10, 6))
plt.plot(eval_epochs, eval_accuracy, marker='o', color='green', label='Evaluation Accuracy')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Evaluation Accuracy', fontsize=18)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# 注意：
# 日志中没有记录训练 accuracy，因此这里只绘制了评估 accuracy 的曲线。
