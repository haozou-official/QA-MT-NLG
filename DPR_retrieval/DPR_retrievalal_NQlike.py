import json
import random
import os
import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRQuestionEncoderTokenizerFast, DPRContextEncoder, DPRContextEncoderTokenizerFast
from datasets import load_dataset
os.environ['TRANSFORMERS_CACHE'] = '/fs/clip-quiz/saptab1/QA-MT-NLG/cache'
os.environ['HF_DATASETS_CACHE'] = '/fs/clip-quiz/saptab1/QA-MT-NLG/cache'
from tqdm import tqdm
from statistics import mode
import sqlite3
import psycopg2
import jsonlines
torch.set_grad_enabled(False)
'''
wiki = []
for line in open('faiss_db_Nov1/faiss_document_store.json', 'r'):
  wiki.append(json.loads(line))
print(wiki[0])

wiki_psgs = {'data' : []}
for i in range(len(wiki)):
  wiki[i]['text'] = wiki[i]['text'].strip()
  wiki_psgs['data'].append(wiki[i])
print(wiki[0])
# save into wiki_psgs.json
with open('faiss_db_Nov1/wiki_psgs.json', 'w') as outfile:
  json.dump(wiki_psgs, outfile)
'''


# faiss_db_Nov1/wiki_dump_embeddings.faiss
# faiss_db_Nov1/faiss_document_store.db
# qanta.train.2018.04.18.json

# embeddings
'''
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
'''
q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# psg, paragraph-level
ds_with_embeddings_avg = load_dataset('json', data_files='faiss_db_Nov1/wiki_psgs.json', field='data', cache_dir='/fs/clip-quiz/saptab1/QA-MT-NLG/cache')
print(ds_with_embeddings_avg)

ds_with_embeddings_avg['train'].load_faiss_index('embeddings', 'faiss_db_Nov1/wiki_dump_embeddings.faiss')
'''
# psg
# ds_with_embeddings = ds.map(lambda example: {'embeddings': ctx_encoder(**ctx_tokenizer(example["text"], return_tensors="pt", max_length=512, truncation=True))[0][0].numpy()})
# ds_with_embeddings['train'].add_faiss_index(column='embeddings')
# ds_with_embeddings['train'].save_faiss_index('embeddings', 'faiss_index/qanta_psgs.faiss')
# --> faiss_db_Nov1/wiki_dump_embeddings.faiss
'''
# questions
retrieved_contexts = []
nqlike_questions_train = []
nqlike_answers_train = []
nqlike_id = []
nqlike_quality_score = []
with jsonlines.open('./Datasets/nq_like_questions_train_with_answer_type_page_phrase_freq_based_quality_scores_sorted.json') as f:
  for line in f.iter():
    nqlike_questions_train.append(line['question_text'])
    nqlike_answers_train.append(line['answer'])
    nqlike_id.append(line['id'])
    nqlike_quality_score.append(line['quality_score'])
print(len(nqlike_questions_train))

f = open('dpr_retrieval/Nov8_retrieval_NQlike.json', 'w')

for i in range(len(nqlike_questions_train)):
  if i%1000==0:
    print(f'{i}/{len(nqlike_questions_train)} done!')
    #print("progress: ", str(i)+"/"+str(len(questions_list))+"\n")
  question = nqlike_questions_train[i]
  question_embedding = q_encoder(**q_tokenizer(question, return_tensors="pt"))[0][0].numpy()
  scores, retrieved_examples = ds_with_embeddings_avg['train'].get_nearest_examples('embeddings', question_embedding, k=3)
  retrieved_result = {}
  retrieved_result['qanta_id'] = nqlike_id[i]
  retrieved_result['question'] = nqlike_questions_train[i]
  retrieved_result['answer'] = nqlike_answers_train[i]
  retrieved_result['quality_score'] = nqlike_quality_score[i]
  retrieved_result['score'] = scores.tolist()
  retrieved_result['retrieved_context'] = retrieved_examples['text']
  #retrieved_contexts.append(retrieved_result)
  json_dict = json.dumps(retrieved_result)
  f.write(json_dict)
  f.write('\n')
f.close()
'''
# save
with open("dpr_retrieval/Nov8_retrieval_NQlike_saved_for_safety.json", 'w') as f:
    for item in retrieved_contexts:
        f.write(json.dumps(item) + "\n")
'''

'''
# open
import json
data = []
with open('dpr_retrieval/Nov5_retrieval_tryout.json') as f:
    for line in f:
        data.append(json.loads(line))
'''
