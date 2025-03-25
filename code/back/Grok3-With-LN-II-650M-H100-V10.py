from transformers import AutoTokenizer, EsmModel, EsmConfig, Trainer, TrainingArguments, IntervalStrategy, EarlyStoppingCallback
from transformers.utils import send_example_telemetry
from transformers.trainer_utils import EvalLoopOutput  # 添加导入 EvalLoopOutput
from datasets import Dataset, ClassLabel, Sequence
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from Bio import SeqIO
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import torch.distributed as dist  # 添加分布式模块导入
import multiprocessing
# import torch.distributed as dist
# dist.barrier()

class CustomEsmClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(0.40)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.out_proj(x)
        return x

class CustomEsmForSequenceClassification(nn.Module):
    def __init__(self, esm_model, config):
        super().__init__()
        self.esm = esm_model
        self.classifier = CustomEsmClassificationHead(config)
        self.config = config

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.esm(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0][:, 0, :]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

class CustomTrainer(Trainer):
    def __init__(self, *args, eval_on_train=True, **kwargs):
        super().__init__(*args, processing_class=kwargs.pop("tokenizer", None), **kwargs)
        self.eval_on_train = eval_on_train

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        outputs = model(**inputs)
        loss = outputs["loss"]
        self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        重写 evaluation_loop，计算损失和预测结果。
        """
        self.model.eval()

        total_eval_loss = 0
        total_samples = 0
        all_predictions = []
        all_labels = []

        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                loss = outputs["loss"]
                logits = outputs["logits"]

            total_eval_loss += loss.detach().float()
            total_samples += inputs["labels"].size(0)

            # 计算预测
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(inputs["labels"].cpu().numpy())

        avg_loss = total_eval_loss / len(dataloader)
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss.item(),
            f"{metric_key_prefix}_accuracy": accuracy_score(all_labels, all_predictions),
            f"{metric_key_prefix}_f1": f1_score(all_labels, all_predictions, average='weighted')
        }

        return EvalLoopOutput(predictions=np.array(all_predictions), label_ids=np.array(all_labels), metrics=metrics, num_samples=total_samples)

    def on_epoch_end(self, args, state, control, **kwargs):
        super().on_epoch_end(args, state, control, **kwargs)
        if self.eval_on_train:
            train_result = self.evaluate(eval_dataset=self.train_dataset, metric_key_prefix="train")
            print(f"Epoch {state.epoch} Training Loss: {train_result['train_loss']:.4f}, Training Accuracy: {train_result['train_accuracy']:.4f}")

    def on_train_end(self, args, state, control, **kwargs):
        super().on_train_end(args, state, control, **kwargs)
        print("Training complete. Running final evaluations...")
        if self.eval_on_train:
            train_result = self.evaluate(eval_dataset=self.train_dataset, metric_key_prefix="train")
            print(f"Final Training Loss: {train_result['train_loss']:.4f}, Training Accuracy: {train_result['train_accuracy']:.4f}")
        eval_result = self.evaluate(eval_dataset=self.eval_dataset, metric_key_prefix="eval")
        print(f"Final Evaluation Loss: {eval_result['eval_loss']:.4f}, Evaluation Accuracy: {eval_result['eval_accuracy']:.4f}")

# Ensure the logging directory exists
log_dir = './logs-650M-6wan-0.01-H100'
os.makedirs(log_dir, exist_ok=True)
print(f"Log directory is set at {log_dir}")

# Read fasta file
def read_fasta_file(fasta_path):
    return [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]

# Read labels and map them to integers
def read_labels(csv_path, label_to_id=None):
    label_df = pd.read_csv(csv_path)
    if label_to_id is None:
        unique_labels = sorted(set(label_df['label']))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
    return label_df['label'].map(label_to_id).tolist(), label_to_id

# Define a function to compute evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": accuracy, "f1": f1}

# Send telemetry data
send_example_telemetry("protein_language_modeling_notebook", framework="pytorch")

# Load data
fasta_sequences = read_fasta_file("All-II-epitope-II-60256.fasta")
labels, label_to_id = read_labels("All-II-epitope-II-60256-Label.csv")

# Split data
train_sequences, test_sequences, train_labels, test_labels = train_test_split(
    fasta_sequences, labels, test_size=0.20, shuffle=True, stratify=labels
)

# Prepare model and tokenizer
model_checkpoint = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
train_tokenized = tokenizer(train_sequences, padding=True, truncation=True, max_length=50)
test_tokenized = tokenizer(test_sequences, padding=True, truncation=True, max_length=50)

# Convert to Dataset format
train_dataset = Dataset.from_dict({'input_ids': train_tokenized['input_ids'], 'attention_mask': train_tokenized['attention_mask'], 'labels': train_labels})
test_dataset = Dataset.from_dict({'input_ids': test_tokenized['input_ids'], 'attention_mask': test_tokenized['attention_mask'], 'labels': test_labels})

# Initialize model
config = EsmConfig.from_pretrained(model_checkpoint, num_labels=len(label_to_id))
esm_model = EsmModel.from_pretrained(model_checkpoint)

# 定义 LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense", "intermediate.dense", "output.dense"],
    lora_dropout=0.05,
    bias="none",
)

# Create the PEFT EsmModel
esm_model = get_peft_model(esm_model, lora_config)

# Create the custom sequence classification model
model = CustomEsmForSequenceClassification(esm_model, config)

# Make sure that the classification head's parameters are trainable
for param in model.classifier.parameters():
    param.requires_grad = True

# 让 Layer Normalization 层可训练
for module in model.esm.base_model.modules():
    if isinstance(module, nn.LayerNorm):
        for param in module.parameters():
            param.requires_grad = True

model_name = model_checkpoint.split("/")[-1]

# Training parameters
args = TrainingArguments(
    output_dir=f"{model_name}-ALL-Epetope-6wan-0.01-H100",
    do_train=True,
    do_eval=True,
    eval_strategy=IntervalStrategy.EPOCH,
    save_strategy=IntervalStrategy.EPOCH,
    learning_rate=1e-5,  # 降低学习率，减少过拟合
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=98,
    weight_decay=0.01,  # 先保持原来的正则化力度
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    deepspeed="ds_config.json",
    push_to_hub=False,
    logging_dir='./logs-650M-6wan-0.01-H100',
    logging_strategy=IntervalStrategy.STEPS,
    logging_steps=10,
    fp16=True,  
    warmup_steps=1000,  # 预热学习率
    lr_scheduler_type="cosine",  # 余弦衰减
    # max_grad_norm=1.0,  # 添加梯度裁剪，限制梯度范数最大值为 1.0
)

# Create Trainer object with early stopping
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=10,
    early_stopping_threshold=0.0001
)

trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    eval_on_train=True,
    callbacks=[early_stopping_callback]
)

# Start training with resource cleanup
try:
    trainer.train()
except Exception as e:
    print(f"Training failed: {e}")
finally:
    if dist.is_initialized():
        dist.destroy_process_group()

# After training, merge the LoRA weights into the EsmModel
model.esm = model.esm.merge_and_unload()

# Save model and tokenizer
model_path = f"{model_name}-ALL-Epetope-6wan-0.01-H100"
torch.save(model.state_dict(), os.path.join(model_path, "pytorch_model.bin"))
tokenizer.save_pretrained(model_path)

# 只有 rank 0 执行评估，避免多进程重复运行
import os
local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 获取 local_rank，默认为 0
if local_rank == 0:
    print("\nLoading saved model and evaluating on training data...")

    # 加载配置和 tokenizer
    config = EsmConfig.from_pretrained(model_checkpoint, num_labels=len(label_to_id))
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 初始化新的 ESM 模型
    esm_model = EsmModel.from_pretrained(model_checkpoint)
    loaded_model = CustomEsmForSequenceClassification(esm_model, config)

    # 加载保存的权重
    model_state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
    loaded_model.load_state_dict(model_state_dict)

    # 将模型移到适当的设备（例如 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)

    # 将模型设置为评估模式
    loaded_model.eval()

    # 创建训练数据的 DataLoader，确保数据被正确转换为张量
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=lambda x: {  # 自定义 collate_fn 确保数据格式
            'input_ids': torch.tensor(x[0]['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(x[0]['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(x[0]['labels'], dtype=torch.long)
        } if isinstance(x, list) and len(x) == 1 else {
            'input_ids': torch.stack([torch.tensor(item['input_ids'], dtype=torch.long) for item in x]),
            'attention_mask': torch.stack([torch.tensor(item['attention_mask'], dtype=torch.long) for item in x]),
            'labels': torch.stack([torch.tensor(item['labels'], dtype=torch.long) for item in x])
        }
    )

    # 初始化变量用于存储预测结果和真实标签
    all_predictions = []
    all_labels = []

    # 在训练数据上进行预测
    with torch.no_grad():
        for batch in train_dataloader:
            # 准备输入数据，确保是张量并移动到设备
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['labels'].to(device)
            }
            
            # 前向传播
            outputs = loaded_model(**inputs)
            logits = outputs['logits']
            
            # 获取预测结果
            predictions = torch.argmax(logits, dim=-1)
            
            # 收集预测和真实标签
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(inputs['labels'].cpu().numpy())

    # 计算评估指标
    train_accuracy = accuracy_score(all_labels, all_predictions)
    train_f1 = f1_score(all_labels, all_predictions, average='weighted')

    # 打印结果
    print(f"Average train accuracy: {train_accuracy:.4f}")
    print(f"Average train F1 score: {train_f1:.4f}")

    # 可选：添加更详细的分类报告
    from sklearn.metrics import classification_report
    print("\nDetailed classification report on training data:")
    print(classification_report(all_labels, all_predictions, target_names=label_to_id.keys()))