import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd

def predict(texts):
    if isinstance(texts, str):
        texts = [texts]

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

    return preds.cpu().numpy()


model_path = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.eval()   # 推理模式

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

df = pd.read_csv('test_new.csv',sep='\t')

id = df['id']
comment = df['comment'].tolist()

preds = predict(comment)

df_result = pd.DataFrame({
    "id": id,
    "pred_label": preds
})

df_result.to_csv("test_pred.csv", index=False)

