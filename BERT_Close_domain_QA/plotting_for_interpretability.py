import re
import string
import json
import sys
import matplotlib.pyplot as plt
# NQlike
system_scores_path_1 = sys.argv[1]
# Nq+NQlike
system_scores_path_2 = sys.argv[2]
system_scores_path_3 = sys.argv[3]

with open(system_scores_path_1) as f:
  scores1 = f.readlines()
print("Num of the EM scores for scores1: ", len(scores1))

for i in range(len(scores1)):
  scores1[i] = float(scores1[i].strip())

samplesMrQA1 = []
for i in range(500, 409501, 500):
  samplesMrQA1.append(i)
#samplesMrQA1 = samplesMrQA1[:-1]
print(len(samplesMrQA1))

for i in range(len(samplesMrQA1)):
  samplesMrQA1[i] = samplesMrQA1[i]*6


with open(system_scores_path_2) as f:
  scores2 = f.readlines()
print("Num of the EM scores for scores2: ", len(scores2))

for i in range(len(scores2)):
  scores2[i] = float(scores2[i].strip())
  #print(scores2[i])
'''
samplesMrQA2 = []
for i in range(500, 60001, 500):
  samplesMrQA2.append(i)
'''
Oct25_checkpoints_validated_path = 'MRQA-NQ+NQ-like-baseline-BERT-base/Oct25_checkpoints_validated.txt'
my_file = open(Oct25_checkpoints_validated_path, 'r')
samplesMrQA2 = my_file.readlines()
my_file.close()
for i in range(len(samplesMrQA2)):
  samplesMrQA2[i] = int(samplesMrQA2[i].strip())
  #print(samplesMrQA2[i])
for i in range(len(samplesMrQA2)):
  samplesMrQA2[i] = samplesMrQA2[i]*6

# NQbaseline
with open(system_scores_path_3) as f:
  scores3 = f.readlines()
print("Num of the EM scores for scores3: ", len(scores3))

for i in range(len(scores3)):
  scores3[i] = float(scores3[i].strip())

samplesMrQA3 = []
for i in range(500, 80001, 500):
  samplesMrQA3.append(i)

#accuracyMrQA = scores1

plt.figure(figsize=(10,10))
fig, ax = plt.subplots()
ax.plot(samplesMrQA1[:], scores1[:], label = 'NQ-like')
ax.plot(samplesMrQA2[:], scores2[:], label = 'NQ+NQ-like')
ax.plot(samplesMrQA3[:], scores3[:], label = 'NQ baseline')

#plt.xticks([samplesMrQA2[0], samplesMrQA2[-1]], visible=True, rotation="horizontal")

plt.title('EM vs Samples for NQ baseline, NQ-like and NQ+NQ-like')
plt.xlabel('Samples') # batches to samples 6
plt.ylabel('EM score')
plt.legend(loc ="upper right")
ax.ticklabel_format(style='plain')
'''
ax.plot(samplesMrQA1[:], scores1[:], label = 'NQ-like')
ax.plot(samplesMrQA2[:], scores2[:], label = 'NQ+NQ-like')
ax.plot(samplesMrQA3[:], scores3[:], label = 'NQ baseline')
'''
save_path = 'Plots_for_interpretability/' + 'Oct25_superimposed' + '.png'
plt.savefig(save_path)
