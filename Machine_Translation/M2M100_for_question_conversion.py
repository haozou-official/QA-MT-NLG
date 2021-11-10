import os
import sys
import random
import re
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from datasets import load_metric
from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M', src_lang="en", tgt_lang="en")

# Sanity Check
# Training

train_tgt_path = '/content/gdrive/Shareddrives/Improving-QA-MT/Dataset/Converted Dataset/Custom Word Transforms Sh v1.1/sh_cwt_v1_1_train_xss.json'
train_src_path = '/content/gdrive/Shareddrives/Improving-QA-MT/Dataset/Quizbowl Dataset/quizdb_train_xss_c.json'

dev_tgt_path = '/content/gdrive/Shareddrives/Improving-QA-MT/Dataset/Converted Dataset/Custom Word Transforms Sh v1.1/sh_cwt_v1_1_dev.json'
dev_src_path = '/content/gdrive/Shareddrives/Improving-QA-MT/Dataset/Quizbowl Dataset/quizdb_dev.json'

test_tgt_path = '/content/gdrive/Shareddrives/Improving-QA-MT/Dataset/Converted Dataset/Custom Word Transforms Sh v1.1/sh_cwt_v1_1_test.json'
test_src_path = '/content/gdrive/Shareddrives/Improving-QA-MT/Dataset/Quizbowl Dataset/quizdb_test_c.json'

train_src_df = pd.read_json(train_src_path, orient='records', lines=True)
train_tgt_df = pd.read_json(train_tgt_path, orient='records', lines=True)

dev_src_df = pd.read_json(dev_src_path, orient='records', lines=True)
dev_tgt_df = pd.read_json(dev_tgt_path, orient='records', lines=True)

test_src_df = pd.read_json(test_src_path, orient='records', lines=True)
test_tgt_df = pd.read_json(test_tgt_path, orient='records', lines=True)

train_src = list(train_src_df.question_text)
train_tgt = list(train_tgt_df.question_text)

dev_src = list(dev_src_df.question_text)
dev_tgt = list(dev_tgt_df.question_text)

test_src = list(test_src_df.question_text)
test_tgt = list(test_tgt_df.question_text)

def remove_points(text):
  patterns = ['For 10 points each, ', 'For 10 points, ',
               'for 10 points, ','for 10 points-', 'For 10 points ',
              'for 10 points ', 'For 10 points',
              'for 10 points', 'For 20 points, '
              'ftp, ', 'FTP, ', 'For ten points, ',
              'FTP ', 'For 10points, ', 'for 10points, ',
              'For 10points', 'for  10  points,  '
              'for ten points, ', 'ftp-',
              ]
  remaining_patterns = ['for 10 points', 'for ten points',
                        'For ten points']
  for pattern in patterns:
    text = re.sub(pattern, '', text)
  for pattern in remaining_patterns:
    text = re.sub(pattern, ' ', text)
  return text

train_src = [remove_points(s) for s in train_src]
dev_src = [remove_points(s) for s in dev_src]
test_src = [remove_points(s) for s in test_src]

class TranslationDataset(Dataset):
  def __init__(self, src, tgt, tokenizer):
    super(TranslationDataset, self).__init__()
    self.src = src
    self.tgt = tgt
    self.tokenizer = tokenizer
    self.max_length = 200

  def __len__(self):
    return len(self.tgt)

  def __getitem__(self, id):
    x = self.src[id]
    y = self.tgt[id]
    input_ids = self.tokenizer.encode(x, return_tensors='pt', padding='max_length',
                                 max_length = self.max_length, truncation=True)
    with self.tokenizer.as_target_tokenizer():
      labels = self.tokenizer(y, return_tensors='pt', padding='max_length',
                         max_length = self.max_length, truncation=True).input_ids
    return {'input_ids':input_ids, 'labels':labels, 'src':x, 'tgt':y}

def my_collate_fn(batch):
  batch_size = len(batch)
  input_ids = batch[0]['input_ids']
  if 'labels' in batch[0].keys():
    labels = batch[0]['labels']
  srcs = [batch[0]['src']]
  tgts = [batch[0]['tgt']]
  for i in range(1,batch_size):
    input_ids = torch.cat((input_ids, batch[i]['input_ids']), dim=0) 
    if 'labels' in batch[i].keys():
      labels = torch.cat((labels, batch[i]['labels']), dim=0)
    srcs.append(batch[i]['src'])
    tgts.append(batch[i]['tgt'])
  return_dict = {'input_ids':input_ids,
                 'labels':labels,
                 'src':srcs,
                 'tgt':tgts}
  return return_dict

train_dataset = TranslationDataset(train_src, train_tgt, tokenizer)
dev_dataset = TranslationDataset(dev_src, dev_tgt, tokenizer)
test_dataset = TranslationDataset(test_src, test_tgt, tokenizer)

trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=my_collate_fn)
devloader = DataLoader(dev_dataset, batch_size=4, collate_fn=my_collate_fn)
testloader = DataLoader(test_dataset, batch_size=4, collate_fn=my_collate_fn)

model = model.to(device)
opt = torch.optim.Adam(model.parameters(), 1e-4)

model.train()
EPOCHS = 4
total_batches = len(trainloader)
for epoch in range(EPOCHS):
  i = 1
  for batch in trainloader:
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    print(f'Epoch [{epoch+1}/{EPOCHS}] Batch [{i}/{total_batches}] Loss = {loss.item()}')
    i += 1
    loss.backward()
    opt.step()
    opt.zero_grad()
