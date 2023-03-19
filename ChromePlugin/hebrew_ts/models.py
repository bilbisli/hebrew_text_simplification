from transformers import AutoTokenizer, BertForTokenClassification, BertForMaskedLM
from transformers import pipeline
from sentence_transformers import SentenceTransformer


def get_mask_model(model_name="imvladikon/alephbertgimmel-base-512"):
    return BertForMaskedLM.from_pretrained(model_name)

def get_ner_model(model_name="jony6484/alephbert-base-finetuned-ner-v2"):
    return BertForTokenClassification.from_pretrained(model_name)

def get_sentence_model(model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    return SentenceTransformer(model_name)

def get_tokenizer(model_name, model):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    NEW_LINE_TOKEN = '<NL>'
    num_added_toks = tokenizer.add_tokens(NEW_LINE_TOKEN)
    model.resize_token_embeddings(len(tokenizer))
    
    return tokenizer

def create_mask_model_pipeline():
    model = get_mask_model()
    tokenizer = get_tokenizer("imvladikon/alephbertgimmel-base-512", model)

    return pipeline("fill-mask", model=model, tokenizer=tokenizer), tokenizer

def create_ner_model_pipeline():
    model = get_ner_model()
    tokenizer = get_tokenizer("jony6484/alephbert-base-finetuned-ner-v2", model)
    
    return pipeline("ner", model=model, tokenizer=tokenizer), tokenizer


sentence_model = get_sentence_model()
mask_model, mask_tokenizer = create_mask_model_pipeline()
ner_model, ner_tokenizer = create_ner_model_pipeline()
