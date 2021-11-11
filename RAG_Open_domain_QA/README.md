# RAG Open Domain QA system

## DPR Retriever
```
RAGretriever = RagRetriever.from_pretrained("facebook/rag-token-nq", 
                                         index_name="legacy")
```

```
  def freeze_embeds(self):
    ''' freeze the positional embedding parameters of the model; adapted from finetune.py '''
    freeze_params(self.model.model.shared)
    for d in [self.model.model.encoder, self.model.model.decoder]:
      freeze_params(d.embed_positions)
      freeze_params(d.embed_tokens)
```

## BART Reader
