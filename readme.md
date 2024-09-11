# Metaphor Detection using BERT models

strategy - used contextulaized embeddings (from pretrained deberta model) with interaction between metaphor word and sentence from start

why deberta - deberta because unlike other bert based encoder models, deberta uses position embeddings in self attention, which has shown to give better results in NLU and NLP tasks

why interact from start - interaction from start can lead to better features creation, creating features manually for a task like metaphor detection would be tough , so we leave it to the model

tokenization - sentencepiece tokenization and input is formated as <s>word in focus</s>sentence</s><pad>
