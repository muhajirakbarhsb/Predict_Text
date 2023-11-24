import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re,string
import emoji
from sklearn.preprocessing import LabelEncoder
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


GOOGLE_DRIVE_FILE_ID = "1xjDyantQr60_jv1vr-xdCkbsJJGD7kxW"

PRE_TRAINED_MODEL_NAME = 'indobenchmark/indobert-base-p2'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

MAX_LEN = 60
BATCH_SIZE = 16


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
    reviews=df.contentp_clean.to_numpy(),
    # targets=df.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

    return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )



def get_predictions(model, data_loader):
    model = model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["review_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
          # targets = d["targets"].to(device)

            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
          )
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
          # real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    # real_values = torch.stack(real_values).cpu()
    # return review_texts, predictions, prediction_probs, real_values
    return review_texts, predictions, prediction_probs

class GPReviewDataset(Dataset):

    def __init__(self, reviews, tokenizer, max_len):
    # def __init__(self, reviews, targets, tokenizer, max_len):
          self.reviews = reviews
      # self.targets = targets
          self.tokenizer = tokenizer
          self.max_len = max_len

    def __len__(self):
          return len(self.reviews)

    def __getitem__(self, item):
          review = str(self.reviews[item])
      # target = self.targets[item]

          encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
          )

          return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            # 'targets': torch.tensor(target, dtype=torch.long)
          }

class SentimentClassifier(nn.Module):

    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        output = self.drop(pooled_output[1])
        return self.out(output)




def predict(teks):
    review_baru = [str(teks)]
    df = pd.DataFrame(review_baru, columns=["contentp_clean"])
    return df


def endpoint(teks):
    df = predict(teks)
    #df = df[['tanggal', 'bundle', 'label', 'is_sentiment', 'contentp_clean']]
    encoder = LabelEncoder()
    encoder.classes_ = np.load('bert_classes.npy', allow_pickle=True)
    model = SentimentClassifier(3)
    model.load_state_dict(torch.load('model/model.bin',  map_location=torch.device('cpu')))
    model = model.to(device)
    testing_data_loader = create_data_loader(df, tokenizer, MAX_LEN, BATCH_SIZE)
    y_review_texts, y_pred, y_pred_probs = get_predictions(
        model,
        testing_data_loader
    )
    ypred = encoder.inverse_transform(y_pred)
    df["Topic_category"] = ypred
    return ypred


def load_model():
	# path to file
	filepath = "model/model.bin"

	# folder exists?
	if not os.path.exists('model'):
		# create folder
		os.mkdir('model')
	
	# file exists?
	if not os.path.exists(filepath):
		# download file
		import gdown
		file_id = "1xjDyantQr60_jv1vr-xdCkbsJJGD7kxW"  # Replace this with your file's ID
		gdown.download(f"https://drive.google.com/uc?id={file_id}&confirm=t", filepath)
    



