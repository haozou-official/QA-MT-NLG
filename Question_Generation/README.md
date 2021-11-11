# Question Generation

## Started With
```
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -r requirements.txt
pip install -e .
```

## check before running
```
import neuralcoref
nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)
```

## Some Heuristics (linguistic perspectives)
* parse tree
* coreference subsitution
* page information frequency table
* answer type propaation
* punctuation check
* sorted based on the quality for generated questions
* consideration for QB last sentence speciality
* name entity recognition
* dealin with synonym
