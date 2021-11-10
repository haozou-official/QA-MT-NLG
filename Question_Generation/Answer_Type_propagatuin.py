import numpy as np
import pandas as pd
import json
import string
import nltk
import time
import os
import re
import random
import argparse
import spacy
import neuralcoref
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

import faiss
from functools import partial
from sklearn.model_selection import train_test_split

print('Downloading NLTK packagaes for Bleu and Meteor Calculation...')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

def clean_chunk(chunk):
  # might have trailing 'and', 'but', etc
  prefixes = ['and', 'but', 'when', 'while', ',']
  punc = ',.'
  chunk = chunk.strip()
  chunk = chunk.strip(punc)
  chunk = chunk.strip()
  chunk = chunk.strip(punc)
  chunk = chunk.strip()
  
  if chunk.endswith(' '):
    chunk = chunk[:-1]
  
  for prefix in prefixes:
    if chunk.startswith(prefix+' '):
      chunk =  chunk[len(prefix)+1:]
    if chunk.endswith(' '+prefix):
      chunk = chunk[:-len(prefix)-1]
  chunk = chunk.strip()

  return chunk 


def parse_tree(df):
  nq_like_questions = {
    'id':[],
    'question_text':[],
    'answer':[]
}

  for index, row in df.iterrows():
    sample = row['question_text'].strip()
    answer = row['answer']
    #sentence_id = row['sentence_id']
    question_id = row['id']
    sample = sample.strip('.')
    
    #if question_id == 2:
    #  break
    
    doc = nlp(sample)
    #print('\033[1m''Actual QB question (After 1st Breakdown):\n', '\033[0m'+sample)
    
    #print('\033[1m''Multiple sentences in question (if any):')
    #print('\033[0m')


    seen = set() # keep track of covered words
    # Find coref clusters
    clusters = doc._.coref_clusters
    #print('CLUSTERS:\n', clusters)
    # Breakdown sentences using Parse Trees
    chunks = []
    for sent in doc.sents:
        conj_heads = [cc for cc in sent.root.children if cc.dep_ == 'conj']
        advcl_heads = [cc for cc in sent.root.children if cc.dep_ == 'advcl']
        #print('Conjuction Heads found :', conj_heads)
        #print('Advcl Heads found :', advcl_heads)
      
        heads = conj_heads + advcl_heads
        for head in heads:
            words = [ww for ww in head.subtree]
            for word in words:
                seen.add(word)

            chunk = (' '.join([ww.text for ww in words]))
            chunks.append( (head.i, chunk) )

        unseen = [ww for ww in sent if ww not in seen]
        chunk = ' '.join([ww.text for ww in unseen])
        chunks.append( (sent.root.i, chunk) )
    
    # Sort the chunks based on word index to ensure first sentences formed come first
    chunks = sorted(chunks, key=lambda x: x[0])
    
    # Ensure no sentences aren't too small
    if len(chunks)>1:
      for idx in range(1, len(chunks)):
        try:
          curr_i, curr_chunk = chunks[idx]
        except:
          #print('idx=',idx)
          #print('chunk len = ', len(chunks))
          raise NotImplementedError
        if len(curr_chunk.split()) < 8 or (curr_chunk.split()[0] in ['after']):
          #print('\nFound a small sent!\n')
          last_i, last_chunk = chunks[idx-1]
          last_chunk = last_chunk + ' ' + curr_chunk
          chunks[idx-1] = (last_i, last_chunk)
          del chunks[idx]
        if (idx+1)>=len(chunks):
          break
      curr_i, curr_chunk = chunks[0]
      if len(curr_chunk.split()) < 8 and len(chunks)>1:
        #print('\nFound a small pre-sent!\n')
        last_i, next_chunk = chunks[1]
        curr_chunk = curr_chunk + ' ' + next_chunk
        chunks[0] = (last_i, curr_chunk)
        del chunks[1]    
    
    # Clean each sentence of trailing and, comma etc
    for i in range(len(chunks)):
      id,chunk = chunks[i]
      chunk = clean_chunk(chunk)
      chunks[i] = (id, chunk)
      
    
    # Coreference subsitution
    pronoun_list = ['he', 'she', 'his', 'her', 'its']
    if len(chunks)>1:
      for i in range(1, len(chunks)):
        curr_i, curr_chunk = chunks[i]
        chunk_doc = nlp(curr_chunk)
        for id, w in enumerate(chunk_doc[:3]):
          #print('Word in chunk doc ', w, '->',w.tag_)
          if w.tag_ in ['NN', 'NNP', 'NNS', 'NNPS']:
            continue
          rep = w.text
          for cluster in clusters:
            #print('Noun chunks: ', cluster[0], '->', [x for x in cluster[0].noun_chunks])
            if (len([x for x in cluster[0].noun_chunks]) > 0) and (str(cluster[0]).lower() not in pronoun_list):
              match_cluster = [str(cc) for cc in cluster]
              #print(match_cluster)
              if w.text in match_cluster:
                rep = match_cluster[0]
                if w.text.lower() in ['his', 'her', 'its', 'it\'s']:
                  rep += '\'s'
                #print(f'Found {w} in cluster!!!')
                #print('Replaceing with ', match_cluster[0])
                break
          if not w.text == rep:
            replacement_list = [str(c) for c in chunk_doc] 
            replacement_list[id] = rep
            curr_chunk = (' ').join(replacement_list)
            chunks[i] = (curr_i, curr_chunk)
          else:
            curr_chunk = '' + curr_chunk


    #print('\033[1m'+'Different nq like statements: (after 2nd breakdown):')
    j = 0
    for ii, chunk in chunks:
      j += 1
      nq_like_questions['question_text'].append(chunk)
      nq_like_questions['id'].append(question_id)
      nq_like_questions['answer'].append(answer)
      #nq_like_questions['sentence_id'].append(sentence_id)
      #nq_like_questions['sub_sentence_id'].append(j)
      if (index % 500) == 0:
        print('\033[0m'+chunk)
  return pd.DataFrame(nq_like_questions)
  
# Transform NQ like sentences into NQ like questions
def uniques( your_string ):    
    words = your_string.split()

    seen = set()
    seen_add = seen.add

    def add(x):
        seen_add(x)  
        return x
    
    output = ' '.join( add(i) for i in words if i not in seen )
    return output

def capitalization(df):
  # capitalize each sentences after parse tree/junk/answer_type extraction and before the transformation
  for idx, row in df.iterrows():
    q = row['question_text']
    q = q[0].upper()+q[1:]
    df.loc[idx, 'question_text'] = q
  return

def last_sent_transform(q):
  if q.split(' ')[:2] == ['name', 'this'] or q.split(' ')[:2] == ['identify', 'this'] or q.split(' ')[:2] == ['give', 'this'] or q.split(' ')[:2] == ['name', 'the'] \
  or q.split(' ')[:2] == ['Name', 'this'] or q.split(' ')[:2] == ['Identify', 'this'] or q.split(' ')[:2] == ['Give', 'this'] or q.split(' ')[:2] == ['Name', 'the'] \
  or q.split(' ')[:2] == ['Give', 'the'] or q.split(' ')[:2] == ['give', 'the']:
    doc = nlp(q)
    tok = []
    flag=0
    for i,token in enumerate(doc[2:6]):
      if token.pos_ == 'NOUN':
        #print('Noun Token = ', token)
        tok.append(str(token))
        flag=1
      else:
        if flag:
          break
    word  = (' ').join(tok)
    
    replacement = 'which is the'
    for k,v in last_sent_word_transform_30000.items():
      if k == 'unk':
        continue
      if word in v:
        replacement = k
        break
    
    transformed_q = q.split(' ')
    transformed_q = transformed_q[2:]
    transformed_q = (' ').join(transformed_q)
    transformed_q = replacement + ' ' + transformed_q   
  elif q.split(' ')[:2] == ['name', 'these'] or q.split(' ')[:2] == ['identify', 'these'] or q.split(' ')[:2] == ['give', 'these'] \
  or q.split(' ')[:2] == ['Name', 'these'] or q.split(' ')[:2] == ['Identify', 'these'] or q.split(' ')[:2] == ['Give', 'these'] \
  or q.split(' ')[:2] == ['Give', 'the'] or q.split(' ')[:2] == ['give', 'the']:
    doc = nlp(q)
    tok = []
    flag=0
    for i,token in enumerate(doc[2:6]):
      if token.pos_ == 'NOUN':
        #print('Noun Token = ', token)
        tok.append(str(token))
        flag=1
      else:
        if flag:
          break
    word  = (' ').join(tok)
    
    replacement = 'which are the'
    for k,v in last_sent_word_transform_30000.items():
      if not k == 'unk':
        continue
      if word in v:
        replacement = k
        break
    transformed_q = q.split(' ')
    transformed_q = transformed_q[2:]
    transformed_q = (' ').join(transformed_q)
    transformed_q = replacement + ' ' + transformed_q   
  else:
    transformed_q = q
  #print('Transformed = ', transformed_q)
  return transformed_q

def remove_duplicates(s):
  words = s.split()
  for i, w in enumerate(words):
    if i >= (len(words)-1):
      continue
    w2 = words[i+1]
    w2 = re.sub('\'s', '', w2)
    if w == w2:
      words = words[:i]+words[i+1:]
  s = " ".join(words)
  return s
  
# use answer_type_dict_before_parse_tree dictionaries for transformation
# pretrained BERT classifier
loaded_model = TFDistilBertForSequenceClassification.from_pretrained("./BERT_Classification/Aug19_answer_type_classification_model/")

def get_answer_type_group(test_sentence):
  predict_input = tokenizer.encode(test_sentence,
                                 truncation=True,
                                 padding=True,
                                 return_tensors="tf")
  tf_output = loaded_model.predict(predict_input)[0]
  tf_prediction = tf.nn.softmax(tf_output, axis=1)
  labels = ['NON_PERSON','PERSON']
  label = tf.argmax(tf_prediction, axis=1)
  label = label.numpy()
  return labels[label[0]]
  
# read json to dict
# v3
with open('./Datasets/QANTA_WITH_ANSWER_TYPE/Answer_Type_Dict_Page_Freq_Based/v3_1/qanta_id_to_the_answer_type_most_freq_phrase_based_on_page_dict.json') as json_file:
    answer_type_dict_before_parse_tree_nq_like_test_v_3 = json.load(json_file)
    
def last_sent_transform(q_with_the_chunks):
  q, q_chunks = junk_last_sentence(q_with_the_chunks)
  if q.split(' ')[:2] == ['name', 'this'] or q.split(' ')[:2] == ['identify', 'this'] or q.split(' ')[:2] == ['give', 'this'] or q.split(' ')[:2] == ['name', 'the'] \
  or q.split(' ')[:2] == ['Name', 'this'] or q.split(' ')[:2] == ['Identify', 'this'] or q.split(' ')[:2] == ['Give', 'this'] or q.split(' ')[:2] == ['Name', 'the'] \
  or q.split(' ')[:2] == ['Give', 'the'] or q.split(' ')[:2] == ['give', 'the']:
    doc = nlp(q)
    tok = []
    flag=0
    for i,token in enumerate(doc[2:6]):
      if token.pos_ == 'NOUN':
        #print('Noun Token = ', token)
        tok.append(str(token))
        flag=1
      else:
        if flag:
          break
    word  = (' ').join(tok)
    
    replacement = 'which is the'
    for k,v in last_sent_word_transform_30000.items():
      if k == 'unk':
        continue
      if word in v:
        replacement = k
        break
    
    transformed_q = q.split(' ')
    transformed_q = transformed_q[2:]
    transformed_q = (' ').join(transformed_q)
    transformed_q = replacement + ' ' + transformed_q   
  elif q.split(' ')[:2] == ['name', 'these'] or q.split(' ')[:2] == ['identify', 'these'] or q.split(' ')[:2] == ['give', 'these'] \
  or q.split(' ')[:2] == ['Name', 'these'] or q.split(' ')[:2] == ['Identify', 'these'] or q.split(' ')[:2] == ['Give', 'these'] \
  or q.split(' ')[:2] == ['Give', 'the'] or q.split(' ')[:2] == ['give', 'the']:
    doc = nlp(q)
    tok = []
    flag=0
    for i,token in enumerate(doc[2:6]):
      if token.pos_ == 'NOUN':
        #print('Noun Token = ', token)
        tok.append(str(token))
        flag=1
      else:
        if flag:
          break
    word  = (' ').join(tok)
    
    replacement = 'which are the'
    for k,v in last_sent_word_transform_30000.items():
      if not k == 'unk':
        continue
      if word in v:
        replacement = k
        break
    transformed_q = q.split(' ')
    transformed_q = transformed_q[2:]
    transformed_q = (' ').join(transformed_q)
    transformed_q = replacement + ' ' + transformed_q   
  else:
    transformed_q = q
  transformed_q = q_chunks+' '+transformed_q
  return transformed_q
  
non_last_sent_transform_dict = {'this ':' which ', 'This ':'Which ',
 'his ':'whose ', 'His ':'Whose ',
'these ':'which ', ''
 'it ':' what ', 'its ': ' what\'s ',
 'It ':'What ', 'Its ':'What\'s ',
    'After ':''
 }

with open('/content/gdrive/Shareddrives/Improving-QA-MT/Colab/Simple Question Transformations/word_transform_dict.json', 'r') as f:
  last_sent_word_transform_30000 = json.load(f)

remove_dict = {
    'For 10 points,  ':'', 'for 10 points,  ':'',
    'For ten points,  ':'', 'for ten points,  ':'',
    'FTP,  ':'', 'ftp,  ':'',
    'For 20 points,  ':'', 'for 20 points,  ':'',
    'For 5 points,  ':'',
    'For 10 points, ':'', 'for 10 points, ':'',
    'For ten points, ':'', 'for ten points, ':'',
    'FTP, ':'', 'ftp, ':'',
    'For 20 points, ':'', 'for 20 points, ':'',
    'For 5 points,':'', 'For 10 points — ':'',
    'For 10 points , ':'', 'for 10 points , ':'',
    'For ten points , ':'', 'for ten points , ':'',
    'FTP , ':'', 'ftp , ':'',
    'For 20 points , ':'', 'for 20 points , ':'',
    'For 5 points , ':'', 
    'For 10 points ':'', 'for 10 points ':'',
    'For ten points ':'', 'for ten points ':'',
    'FTP ':'', 'ftp ':'',
    'For 20 points ':'', 'for 20 points ':'',
    'For 5 points ':''
}

def capitalization(df):
  # capitalize each sentences after parse tree/junk/answer_type extraction and before the transformation
  for idx, row in df.iterrows():
    q = row['question_text']
    if len(q)>2:
      q = q[0].upper()+q[1:]
    df.loc[idx, 'question_text'] = q
  return

# Aug24 new version transformation v3_2_2/v2_1_2
# Aug24: 1. replace only first pronoun instead of all
# 2. this/This special DET cases: do not add NOUN behind them since they already have one (just simply this->which)
# 3. ill-formed sentences: is the first token is VERB and it is not the 'said to be' cases, then do not add is/are in front of it
# 4. keep the chunks before ftps (last sentence) not merge it into previous sentence
# 5. punctuation issue: to make nq_like a question (fix ill punctuations at the end of the sentences; add question marks)
def transformation(df):
  # junk chunks before 'FTP's
  #junk_last_sentence(df)
  # find the answer type dict from last sentence (already extracted the answer type dictionary)
  answer_type_dict = answer_type_dict_before_parse_tree_nq_like_test_v_3
  # capitalize the sentences after the answer_type extraction [Aug23: and deal with no pronous cases]
  capitalization(df)
  for idx, row in df.iterrows():
    if (idx+1)%5000==0:
      print(f'{idx+1} samples done!')
    q = row['question_text']
    qb_id = str(row['id']) # match the answer type from answer_type_dict
    if idx < (len(df)-1):
      if df.loc[idx+1, 'id'] == row['id']:
        # intermediate sentences'id
        if qb_id in answer_type_dict.keys():
          answer_type = answer_type_dict[qb_id] # get the answer type from qb_id
          
          # detect if the answer_type (noun) is a person or a thing
          if answer_type in last_sent_word_transform_30000['who is the']:
            # answer_type is PERSON
            replacement_prefix = 'which'
            replacement = replacement_prefix+' '+answer_type
            q_orig = q
            # he/He/he's/He's/his/His/who/Who/whose/Whose
            for k in ['He ', 'Who ', 'She ']:
              q = re.sub(k, replacement+' ', q, 1)
            for k in ['This ']:
              q = re.sub(k, 'Which ', q, 1)
            for k in [' he ', ' who ', ' she ', ' him ']:
              q = re.sub(k, ' '+replacement+' ', q, 1)
            for k in [' this ']:
              q = re.sub(k, ' '+' which ', q, 1)         
            for k in ['He\'s ', 'His ', 'Whose ', 'She\'s ', 'Her ']:
              q = re.sub(k, replacement+'\'s'+' ', q, 1)        
            for k in [' he\'s ', ' his ', ' whose ', ' she\'s ', ' her ']:
              q = re.sub(k, ' '+replacement+'\'s'+' ', q, 1)
          else:
            # classified as PERSON by BERT
            classification_output = get_answer_type_group(answer_type)
            if classification_output == 'PERSON':
              # answer_type is PERSON
              replacement_prefix = 'which'
              replacement = replacement_prefix+' '+answer_type
              # he/He/he's/He's/his/His/who/Who/whose/Whose
              for k in ['He ', 'Who ', 'She ']:
                q = re.sub(k, replacement+' ', q, 1)
              for k in ['This ']:
                q = re.sub(k, 'Which ', q, 1)      
              for k in [' he ', ' who ', ' she ', ' him ']:
                q = re.sub(k, ' '+replacement+' ', q, 1)
              for k in [' this ']:
                q = re.sub(k, ' which ', q, 1)
              for k in ['He\'s ', 'His ', 'Whose ', 'She\'s ', 'Her ']:
                q = re.sub(k, replacement+'\'s'+' ', q, 1)
              for k in [' he\'s ', ' his ', ' whose ', ' she\'s ', ' her ']:
                q = re.sub(k, ' '+replacement+'\'s'+' ', q, 1)
            else:
              # answer_type is a thing 
              replacement_prefix = 'which'
              replacement = replacement_prefix+' '+answer_type
              # swap in with the replacement
              # what/What/what's/What's/it/It/it's/It's/its/Its -> what/What+replacement
              for k in ['What ', 'It ']:
                q = re.sub(k, replacement+' ', q, 1)
              for k in ['This ']:
                q = re.sub(k, 'Which ', q, 1)
              for k in [' what ', ' it ']:
                q = re.sub(k, ' '+replacement+' ', q, 1)
              for k in [' this ']:
                q = re.sub(k, ' which ', q, 1)               
              for k in ['What\'s ', 'Its ', 'It\'s ']:
                q = re.sub(k, replacement+'\'s'+' ', q, 1)
              for k in [' what\'s ', ' its ', ' it\'s ']:
                q = re.sub(k, ' '+replacement+'\'s'+' ', q, 1)

        else:
            for k,v in non_last_sent_transform_dict.items():
              q = re.sub(' '+k, ' '+v, q, 1)
              if q.startswith(k):
                q = v + q[len(k):]
      else:
        # last sentence
        # for k,v in remove_dict.items():
        #   q = re.sub(k, v, q)
        q = last_sent_transform(q)
    else:
      # last sentence
      # for k,v in remove_dict.items():
      #   q = re.sub(k, v, q)
      q = last_sent_transform(q)
    # remove adjancent duplicates
    q = remove_duplicates(q)
    # capitalize the first letter of each sentence
    # q = q[0].upper()+q[1:]
    df.loc[idx, 'question_text'] = q
  return 

def deal_with_no_pronouns_cases(df):
  # make the first letter lower case
  
  # input: df after the parse tree steps and before transformation
  for idx, row in df.iterrows():
    q = row['question_text']
    if len(q)>2:
      q = q[0].lower()+q[1:]
      # qb_id = str(row['id'])
      # answer_type_dict = answer_type_dict_before_parse_tree_nq_like_v2_1
      # answer_type = answer_type_dict[qb_id]
      # processed_text = nlp(answer_type)
      # lemma_tags = {"NNS", "NNPS"}

      question_test = nlp(q)
      pronouns_tags = {"PRON", "WDT", "WP", "WP$", "WRB", "VEZ"}
      # check whether there are any pronouns or not in the sentence q
      flag = True
      for token in question_test:
        if token.tag_ in pronouns_tags:
          flag = False
          break
      
      if flag == True:
        # no pronouns in the question

        # check wether answer type is singular or plural
        qb_id = str(row['id'])
        #print(qb_id)
        answer_type_dict = answer_type_dict_before_parse_tree_nq_like_test_v_3
        if qb_id in answer_type_dict.keys():
          answer_type = answer_type_dict[qb_id]
        #answer_type = answer_type_dict[qb_id]
        #print(answer_type)
          processed_text = nlp(answer_type)
          lemma_tags = {"NNS", "NNPS"}

          sigular_plural_flags = True # singular
          for token in processed_text:
            if token.tag_ == 'NNPS':
              sigular_plural_flags = False # plural
              break
          
          # check if the first toke is VERB
          if len(question_test) > 3:
            if question_test[0].pos_ == 'VERB' and question_test[1].pos_ != 'PART' and question_test[2].pos_ != 'AUX':
              replacement = 'which '+answer_type+' '
              q = replacement+q
            else:
              if sigular_plural_flags == False:
                # plural
                replacement = 'which '+answer_type+' are '
                q = replacement+q  
              else:
                # singular
                replacement = 'which '+answer_type+' is '
                q = replacement+q
    # capitalize the first letter of each sentence
      q = q[0].upper()+q[1:]
    df.loc[idx, 'question_text'] = q
  return

# def punctuation_check(df):
#   punctuations = '''!()-[]{};:'"\,<>./@#$%^&*_~'''
#   for idx, row in df.iterrows():
#     q = row['question_text']
#     token_list = q.split()
#     end_token = token_list[len(token_list)-1]
#     if end_token in punctuations:
#       q = ' '.join(token_list[:-1])
#       q = q+' ?'
#     elif end_token == '?':
#       q = q
#     else:
#       q = q+' ?'
#     df.loc[idx, 'question_text'] = q
#   return

# steps: parse_tree, transformation, deal_with_no_pronouns_cases, puctuation_check

# def deal_with_synonym(k, q):
#   noun_phrase = extract_noun_phrase_after_pronoun(k, q)
#   if noun_phrase != None:

# def deal_with_punctuation(df):
# # punctuation at the end; question mark

# def transformation_pipeline(df):
#   nq_like_questions = parse_tree(df)
#   transformation(nq_like_questions)
#   deal_with_no_pronouns_cases(nq_like_questions)
#   punctuation_check(nq_like_questions)
#   return nq_like_questions

def scale_up():
    train = '/content/gdrive/Shareddrives/Improving-QA-MT/Colab/Scripts/Datasets/Post EMNLP/Parse Tree/NQ_like_statements_our_qanta_train.json'
    dev = '/content/gdrive/Shareddrives/Improving-QA-MT/Colab/Scripts/Datasets/Post EMNLP/Parse Tree/NQ_like_statements_our_qanta_dev.json'
    test = '/content/gdrive/Shareddrives/Improving-QA-MT/Colab/Scripts/Datasets/Post EMNLP/Parse Tree/NQ_like_statements_our_qanta_test.json'
    train = pd.read_json(train, lines=True, orient='records')
    dev = pd.read_json(dev, lines=True, orient='records')
    test = pd.read_json(test, lines=True, orient='records')
    test = test[['qanta_ids', 'questions', 'sentence_ids', 'sub_sentence_ids', 'answers']]
    dev = dev[['qanta_ids', 'questions', 'sentence_ids', 'sub_sentence_ids', 'answers']]
    train = train[['qanta_ids', 'questions', 'sentence_ids', 'sub_sentence_ids', 'answers']]
    test = test.rename(columns={'questions':'question_text'})
    dev = dev.rename(columns={'questions':'question_text'})
    train = train.rename(columns={'questions':'question_text'})
    test = test.rename(columns={'answers':'answer'})
    dev = dev.rename(columns={'answers':'answer'})
    train = train.rename(columns={'answers':'answer'})
    test = test.rename(columns={'qanta_ids':'id'})
    dev = dev.rename(columns={'qanta_ids':'id'})
    train = train.rename(columns={'qanta_ids':'id'})
    deal_with_no_pronouns_cases(nq_like_train)
    
    dataset1_lst = []
    for i in range(len(nq_like_train)):
        dataset1 = {}
        dataset1['id'] = str(nq_like_train.iloc[i]['id'])
        dataset1['question_text'] = nq_like_train.iloc[i]['question_text']
        dataset1['answer'] = nq_like_train.iloc[i]['answer']
        dataset1_lst.append(dataset1)
    with open("Datasets/QANTA_WITH_ANSWER_TYPE/Rule_based_NQ_like_POST_EMNLP/nq_like_questions_train_with_answer_type_page_phrase_freq_based_sub_"+str(sub_num)+".json", 'w') as f:
        for item in dataset1_lst:
            f.write(json.dumps(item) + "\n")