## validation starts from here
import torch
import os
import sys
os.environ['TRANSFORMERS_CACHE'] = '/fs/clip-quiz/saptab1/QA-MT-NLG/MRQA-NQ+NQ-like-baseline-BERT-base/cache'
os.environ['HF_DATASETS_CACHE'] = '/fs/clip-quiz/saptab1/QA-MT-NLG/MRQA-NQ+NQ-like-baseline-BERT-base/cache'
import pandas as pd
import numpy as np
import json
import random
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import datasets
from datasets import load_dataset,ClassLabel, Sequence
from transformers import AutoTokenizer
import collections
from tqdm.auto import tqdm
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import default_data_collator
import nltk

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using Device = ', device)

nltk.download('punkt')

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
max_length = 380 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
pad_on_right = tokenizer.padding_side == "right"

# Data <Please change the dataset names appropriately>
#tiny_train_data = load_dataset('json', data_files='../Datasets/NaturalQuestions_train.json', split='train[:]')
# 2 different variables, so does not matter if train split is used
tiny_val_data = load_dataset('json', data_files='../Datasets/NaturalQuestions_dev.json', split='train[:]')

def prepare_validation_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
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

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["qid"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

tiny_val_tokenized_datasets = tiny_val_data.map(prepare_validation_features, batched=True, batch_size=64,
                                    remove_columns=['context', 'detected_answers', 'answers',
                                                    'question', 'qid'])

def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size = 20, max_answer_length = 30):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["qid"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    #print(f"Post-processing {len(examples)} example predictions split into {len(features)} features")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []
        
        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}
        
        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if not squad_v2:
            predictions[example["qid"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["qid"]] = answer

    return predictions

squad_v2 = False

def exact_match(predictions, references):
  match = 0
  corr = []
  prev_match = 0
  for rs,p in zip(references, predictions):
    for r in rs:
      r_tok = nltk.word_tokenize(r)
      p_tok = nltk.word_tokenize(p)
      if r_tok == p_tok:
        match += 1
        break
    if match != prev_match:
      corr.append((rs, p))
      prev_match = match 
  #print('Match = ',match, ' Out of ', len(predictions))
  return match/len(predictions), corr

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
max_length = 380 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
pad_on_right = tokenizer.padding_side == "right"

# Data <Please change the dataset names appropriately>

def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
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

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
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
        answers = examples["answers"][sample_index]
        detected_answers = examples['detected_answers'][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(detected_answers["char_spans"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            # start_char = detected_answers["char_spans"][0]['start'][0]
            # end_char = detected_answers["char_spans"][0]['end'][0]
            start_char = detected_answers["char_spans"][0][0]
            end_char = detected_answers["char_spans"][0][1]
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

train_batch_size = 6
eval_batch_size = 8
args = TrainingArguments(
    f"Oct24_MRQA_NQ_NQlike_Closed_Domain_QA_BERTbase_seq_plot",
    evaluation_strategy = "steps",
    learning_rate=2e-5,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_steps = 2000
)

'''
tiny_train_tokenized_datasets = tiny_train_data.map(prepare_train_features, batched=True, batch_size=64,
                                    remove_columns=['context', 'detected_answers', 'answers',
                                                    'question', 'qid'])
'''
tiny_val_tokenized_datasets = tiny_val_data.map(prepare_validation_features, batched=True, batch_size=64,
                                    remove_columns=['context', 'detected_answers', 'answers',
                                                    'question', 'qid'])

data_collator = default_data_collator

nq_initial_checkpoint = int(sys.argv[1])
nq_total_checkpoints = int(sys.argv[2])
nqlike_initial_checkpoint = int(sys.argv[3])
nqlike_total_checkpoints = int(sys.argv[4])

#periodicity = int(sys.argv[3])
scores = []

#EPOCHS = 3
#for epoch in EPOCHS

for i in range(nq_initial_checkpoint, nq_total_checkpoints, 1):
  model_path = 'Tryout_Oct24_MRQA_NQ_NQlike_Closed_Domain_QA_BERTbase_seq/epoch_2_step_'+str(i)+'_NQorig'
  flag = os.path.exists(model_path)
  if (flag==False):
    #print(model_path)
    continue
  model = AutoModelForQuestionAnswering.from_pretrained(model_path)

  
  trainer = Trainer(
      model,
      args,
      train_dataset=tiny_val_tokenized_datasets,
      eval_dataset=tiny_val_tokenized_datasets,
      data_collator=data_collator,
      tokenizer=tokenizer,
  )

  raw_predictions = trainer.predict(tiny_val_tokenized_datasets)
  final_predictions = postprocess_qa_predictions(tiny_val_data, tiny_val_tokenized_datasets, raw_predictions.predictions)
  if squad_v2:
      formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in final_predictions.items()]
  else:
      formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
  references = [{"id": ex["qid"], "answers": ex["answers"]} for ex in tiny_val_data]

  pred_answers = []
  for p in formatted_predictions:
    pred_answers.append(p['prediction_text'])
  #print(len(pred_answers))

  gold_answers = []
  for r in references:
    gold_answers.append(r['answers'])
  #print(len(gold_answers))

  dataset = []
  for j in range(len(pred_answers)):
    data = {}
    data['pred'] = pred_answers[j]
    data['gold'] = gold_answers[j]
    dataset.append(data)
  # save
  with open('Oct24_MRQA_NQ_NQlike_Closed_Domain_QA_BERTbase_seq_predictions/NQorig_epoch2_step_'+str(i)+'_predictions_and_references.json', 'w') as f:
      for item in dataset:
         f.write(json.dumps(item)+"\n")

  score, correct = exact_match(pred_answers, gold_answers)
  # print(score)
  scores.append(score)

for i in range(nqlike_initial_checkpoint, nqlike_total_checkpoints, 1):
  model_path = 'Tryout_Oct24_MRQA_NQ_NQlike_Closed_Domain_QA_BERTbase_seq/epoch_2_step_'+str(i)+'_NQlike'
  flag = os.path.exists(model_path)
  if (flag==False):
    #print(model_path)
    continue
  model = AutoModelForQuestionAnswering.from_pretrained(model_path)


  trainer = Trainer(
      model,
      args,
      train_dataset=tiny_val_tokenized_datasets,
      eval_dataset=tiny_val_tokenized_datasets,
      data_collator=data_collator,
      tokenizer=tokenizer,
  )

  raw_predictions = trainer.predict(tiny_val_tokenized_datasets)
  final_predictions = postprocess_qa_predictions(tiny_val_data, tiny_val_tokenized_datasets, raw_predictions.predictions)
  if squad_v2:
      formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in final_predictions.items()]
  else:
      formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
  references = [{"id": ex["qid"], "answers": ex["answers"]} for ex in tiny_val_data]

  pred_answers = []
  for p in formatted_predictions:
    pred_answers.append(p['prediction_text'])
  #print(len(pred_answers))

  gold_answers = []
  for r in references:
    gold_answers.append(r['answers'])
  #print(len(gold_answers))

  dataset = []
  for j in range(len(pred_answers)):
    data = {}
    data['pred'] = pred_answers[j]
    data['gold'] = gold_answers[j]
    dataset.append(data)
  # save
  with open('Oct24_MRQA_NQ_NQlike_Closed_Domain_QA_BERTbase_seq_predictions/NQlike_epoch2_step_'+str(i)+'_predictions_and_references.json', 'w') as f:
      for item in dataset:
         f.write(json.dumps(item)+"\n")

  score, correct = exact_match(pred_answers, gold_answers)
  # print(score)
  scores.append(score)



with open('Nov1_MRQA_NQ_NQlike_Closed_Domain_QA_BERTbase_seq_epoch2.txt', 'w') as f:
    for item in scores:
        f.write("%s\n" % item)
