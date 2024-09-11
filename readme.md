# Metaphor Detection using BERT-based models

# Strategy
Used contextulaized embeddings (from pretrained deberta model) with interaction between metaphor word and sentence from start

# Why DeBERTa 
DeBERTa unlike other bert based encoder models,uses position embeddings in self attention, which has shown to give better results in NLU and NLP tasks

# Why interact from start
Interaction from start can lead to better features creation, creating features manually for a task like metaphor detection would be tough , so we leave it to the model

# Tokenization
sentencepiece tokenization and input is formated as <s>word in focus</s>sentence</s><pad>

# Environment
Python: >= 3.6 PyTorch: >= 1.5.0
