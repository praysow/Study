import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 로드
path = 'c:/_data/kaggle/suhak/'
train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
sample = pd.read_csv(path + 'sample_submission.csv')

# 경로(Path) 및 라벨 인코더(Label Encoder) 설정
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 라벨 인코딩
label_encoder = LabelEncoder()
train['answer'] = label_encoder.fit_transform(train['answer'])

# 데이터 전처리 및 DataLoader 생성
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)  # 라벨 값을 수정하여 적용
        }

max_len = 128
ba = 40
train_texts, val_texts, train_labels, val_labels = train_test_split(train['problem'], train['answer'], test_size=0.2)
train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_len)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_len)
train_loader = DataLoader(train_dataset, batch_size=ba, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=ba, shuffle=False)

# 트랜스포머 모델 초기화 및 옵티마이저 설정
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=train['answer'].nunique())
optimizer = AdamW(model.parameters(), lr=0.0000000000000001)

# 학습
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for epoch in range(100):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    
    model.eval()
    val_accuracy = []
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean().item()
        val_accuracy.append(accuracy)
    
    print(f'Epoch {epoch+1}, Validation Accuracy: {sum(val_accuracy)/len(val_accuracy)}')

# 테스트 데이터 예측 및 저장
test_texts = test['problem']
test_dataset = CustomDataset(test_texts, [0]*len(test_texts), tokenizer, max_len)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model.eval()
predictions = []
for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    predictions.extend(preds)

# 예측 결과를 원래의 라벨로 디코딩하여 저장
# sample['answer'] = label_encoder.inverse_transform(predictions)
# sample.to_csv(path + "transformer_predictions.csv", index=False)

print("Inference complete. Predictions saved.")
