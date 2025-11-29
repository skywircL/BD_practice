import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import re
import jieba
from peft import LoraConfig, TaskType, get_peft_model

tokenizer = AutoTokenizer.from_pretrained('./bert')
model = AutoModelForSequenceClassification.from_pretrained('./bert')

config = LoraConfig(
    task_type=TaskType.SEQ_CLS, # 任务类型，对于序列分类任务，应设置为 SEQ_CLS
    inference_mode=False,     # 设置为 False，因为我们正在进行训练
    r=16,                     # LoRA 的秩，是关键超参数。常用的值有 8, 16, 32。r 越大，可训练参数越多，表达能力越强，但计算成本也越高。
    lora_alpha=32,            # LoRA 的缩放因子，通常设置为 r 的两倍。
    lora_dropout=0.1,         # LoRA 层的 Dropout 概率。
    target_modules=["Wqkv", "Wo"], # 指定要应用 LoRA 的模块。对于 BERT 模型，通常是 "query" 和 "value"。可以通过 print(model) 查看模型结构来确定。
    bias="none"               # 是否训练偏置项。"none" 表示不训练。
)
peft_model = get_peft_model(model, config)
print("\nLoRA 模型可训练参数:")
peft_model.print_trainable_parameters()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=128
        )
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    macro_f1 = f1_score(labels, predictions, average='macro')
    micro_f1 = f1_score(labels, predictions, average='micro')
    acc = accuracy_score(labels, predictions)

    return {
        "eval_macro_f1": macro_f1,  # 必须加上这个 key
        "eval_micro_f1": micro_f1,
        "eval_accuracy": acc,
    }


df = pd.read_csv('train.csv',sep='\t')
# 进行数据清洗
def clean_text(text):
    # 只删掉 @ # $ % ^ & * ( ) - _ + = [ ] { } | \ / < > ~ 《》 「」 etc.
    # 但保留 ，。！？；：“”‘’和所有表情
    regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9，。！？；：“”‘’\u3000-\u301f\u3040-\u318f\u31a0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]')
    text = regex.sub(' ', text)
    return text.strip()
df['comment'] = df['comment'].astype(str).apply(clean_text)

train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
    df['comment'], df['label'], test_size=1000, stratify=df['label'], random_state=42)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_val_texts, train_val_labels, test_size=1000, stratify=train_val_labels, random_state=42)



train_dataset = Dataset(train_texts.reset_index(drop=True), train_labels.reset_index(drop=True),tokenizer)
val_dataset = Dataset(val_texts.reset_index(drop=True), val_labels.reset_index(drop=True),tokenizer)



training_args = TrainingArguments(
    output_dir='./my_food_safety_model',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=20,
    save_strategy="epoch",
    load_best_model_at_end=True,
    eval_strategy="epoch",
    metric_for_best_model="eval_macro_f1",

    learning_rate=3e-5,
    warmup_steps=200,
    weight_decay=1e-4,
    bf16=True,
    dataloader_num_workers=0,
    report_to="none",                  # 不推送到 wandb
    torch_compile=False,

)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./model")
tokenizer.save_pretrained("./model")

