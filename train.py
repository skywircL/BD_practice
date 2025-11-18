import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer
import pandas as pd
from sklearn.model_selection import train_test_split



tokenizer = AutoTokenizer.from_pretrained('./bert')
model = AutoModelForSequenceClassification.from_pretrained('./bert')

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


df = pd.read_csv('train.csv',sep='\t')


train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
    df['comment'], df['label'], test_size=1000, stratify=df['label'], random_state=42)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_val_texts, train_val_labels, test_size=1000, stratify=train_val_labels, random_state=42)



train_dataset = Dataset(train_texts.reset_index(drop=True), train_labels.reset_index(drop=True))
val_dataset = Dataset(val_texts.reset_index(drop=True), val_labels.reset_index(drop=True))



training_args = TrainingArguments(
    output_dir='./my_food_safety_model',
    num_train_epochs=3,
    per_device_train_batch_size=64,
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
    fp16=True,
    dataloader_num_workers=8,
    report_to="none",                  # 不推送到 wandb
    torch_compile=True,

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
trainer.save_model("./model")
tokenizer.save_pretrained("./model")

