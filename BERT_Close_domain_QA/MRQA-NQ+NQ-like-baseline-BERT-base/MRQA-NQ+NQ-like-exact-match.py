import nltk
import re
import string
import json
import sys

nltk.download('punkt')
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


initial_checkpoint = int(sys.argv[1])
final_checkpoint = int(sys.argv[2])
periodicity = int(sys.argv[3])

scores = []

for i in range(initial_checkpoint, final_checkpoint, periodicity):
  golds = []
  preds = []

  with open('Oct18_MRQA_NQ_NQlike_Closed_Domain_QA_2k_BERTbase_predictions/predictions_and_references_'+str(i)+'.txt', 'r') as f:
     lines = f.readlines()
     for line in lines:
       gold = re.findall('\[(.*?)\]', line)
       pattern = "'pred':\(.*?\)\, 'gold'"
       pattern = "'pred': (.*?), 'gold'"
       pred = re.search(pattern, line).group(1)
       #print(pred)
       # Take first instance in re list as there is only one instance
       #print(gold)
       #gold = gold[0]
       #print(gold)
       #print(type(gold))
       # strip strings of extra quotation marks due to regex code
       #gold = [g[1:-1] for g in gold]
       #print(gold)
       golds.append(gold)
       preds.append(pred)
       #break
  #break
  #print('Golds -> ', golds)
  #print(type(golds))
  #print('Preds ->', preds)
  #print(type(preds))
  score,_ = exact_match(preds, golds)
  print(score)
  scores.append(score)

save_path = 'exact_match_score_' + str(initial_checkpoint) + '_' + str(final_checkpoint) + '.txt'
with open(save_path, 'w') as f:
  for s in scores:
    f.write("%s\n" %s)

