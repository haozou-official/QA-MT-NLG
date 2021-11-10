#code
import sys
import os
os.environ['TRANSFORMERS_CACHE'] = '/fs/clip-quiz/saptab1/QA-MT-NLG/MRQA-NQ+NQ-like-baseline-BERT-base/cache'
os.environ['HF_DATASETS_CACHE'] = '/fs/clip-quiz/saptab1/QA-MT-NLG/MRQA-NQ+NQ-like-baseline-BERT-base/cache'
import pandas as pd
import numpy as np
import json
import random
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import datasets
from datasets import load_dataset,ClassLabel, Sequence
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, default_data_collator, Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device is: ",device)

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
max_length = 380 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
pad_on_right = tokenizer.padding_side == "right"

global cumulative_size
cumulative_size =  0
tokenized_examples_list = []
cumulative_tokenized_examples_list = []

nqlike_train_path = sys.argv[1]

'''
# reformat
tiny_train_data = []
tiny_val_data = []
with open(nq_train_path) as f:
    for line in f:
        tiny_train_data.append(json.loads(line))

with open(nq_dev_path) as f:
    for line in f:
        tiny_val_data.append(json.loads(line))

for i in range(len(tiny_val_data)):
  tiny_val_data[i]['char_spans'] = tiny_val_data[i]['detected_answers']['char_spans']
  tiny_val_data[i]['qanta_id'] = i
  tiny_val_data[i]['quality_score'] = 1.0
  tiny_val_data[i]['answer'] = tiny_val_data[i]['answers'][0]


for i in range(len(tiny_train_data)):
  tiny_train_data[i]['char_spans'] = tiny_train_data[i]['detected_answers']['char_spans']
  tiny_train_data[i]['qanta_id'] = i
  tiny_train_data[i]['quality_score'] = 1.0
  tiny_train_data[i]['answer'] = tiny_train_data[i]['answers'][0]

for i in range(len(tiny_val_data)):
  del tiny_val_data[i]['detected_answers']
  del tiny_val_data[i]['qid']
  del tiny_val_data[i]['answers']

for i in range(len(tiny_train_data)):
  del tiny_train_data[i]['detected_answers']
  del tiny_train_data[i]['qid']
  del tiny_train_data[i]['answers']


with open("/fs/clip-quiz/saptab1/QA-MT-NLG/Datasets/NaturalQuestions_train_reformatted.json", 'w') as f:
    for item in tiny_train_data:
        f.write(json.dumps(item) + "\n")

with open("/fs/clip-quiz/saptab1/QA-MT-NLG/Datasets/NaturalQuestions_dev_reformatted.json", 'w') as f:
    for item in tiny_val_data:
        f.write(json.dumps(item) + "\n")


print('Loading validation data...')
tiny_val_data = load_dataset('json', data_files=nq_dev_path, split='train[:]')
print('Loaded validation data!')
'''
print('Loading Training data...')
# Data <Please change the dataset names appropriately>
tiny_nqlike_train_data = load_dataset('json', data_files=nqlike_train_path, split='train[:]')

print('Loaded Training Data!')
# 2 different variables, so does not matter if train split is used

print('Dataset Loaded!')

def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation f the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    #tokenized_examples_list.append(len(tokenized_examples))

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    #print(sample_mapping)
    tokenized_examples_list.append(len(sample_mapping))
    
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        #answers = examples["answers"][sample_index]
        answers = examples["answer"][sample_index]
        #print(examples['char_spans'])
        spans = examples['char_spans'][sample_index]
        #if i == 0:
            #print(spans)
        #detected_answers = examples['detected_answers'][sample_index]
        # If no answers are given, set the cls_index as answer.
        if spans[0][0] == 0 and spans[0][1] == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            # start_char = detected_answers["char_spans"][0]['start'][0]
            # end_char = detected_answers["char_spans"][0]['end'][0]
            start_char = spans[0][0]
            end_char = spans[0][1]
            #start_char = answers["answer_start"][0]
            #end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

        # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1
            #print(token_end_index)
            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while token_end_index > 0 and offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    #print(len(tokenized_examples['input_ids']))
    #print(len(tokenized_examples['attention_mask']))

    #print(tokenized_examples['attention_mask']])
    #return_dict = {'input_ids':tokenized_examples['input_ids'],
     #              'start_positions':tokenized_examples['start_positions'],
     #              'end_positions':tokenized_examples['end_positions']}
    return tokenized_examples	

print('Tokenizing Train Data...')

tiny_nqlike_train_tokenized_datasets = tiny_nqlike_train_data.map(prepare_train_features, batched=True, batch_size=1,
                                    remove_columns=['context', 'char_spans', 'answer',
                                                    'question', 'qanta_id', 'quality_score'])

print('Tokenized NQlike Train Data!')
'''
print('Tokenizing Val Data...')
tiny_val_tokenized_datasets = tiny_val_data.map(prepare_train_features, batched=True, batch_size=64,
                                    remove_columns=['context', 'char_spans', 'answer',
                                                    'question', 'qanta_id', 'quality_score'])

print('Tokenized Val Data!')
'''
#train_batch_size = 6
#eval_batch_size = 8

#nqlike_trainloader = torch.utils.data.Dataloader(tiny_nqlike_train_tokenized_datasets, batchsize=train_batch_size, shuffle=False)
# save

for element in tokenized_examples_list:
   cumulative_size += element
   cumulative_tokenized_examples_list.append(cumulative_size)

with open("./mapping.txt", 'w') as f:
  for (item1, item2) in zip(tokenized_examples_list,cumulative_tokenized_examples_list):
    f.write(str(item1))
    f.write(' ')
    f.write(str(item2))
    f.write('\n')


