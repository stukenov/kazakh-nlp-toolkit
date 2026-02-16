import torch
import numpy as np
import transformers
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "ai-forever/mGPT-1.3B-kazakh"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

texts = [
    
    "Александр Сергеевич Пушкин дүниеге келген",
    "Қазақстан Республикасының құрамындағы ең үлкен қала",
]

model.eval()
transformers.set_seed(1337)
for text in texts:
    input_ids = tokenizer.encode(text, return_tensors="pt")
    out = model.generate(
        input_ids, 
        min_length=100,
        max_length=100,
        eos_token_id=5, 
        pad_token=1,
        do_sample=True,
        top_k=0,
        top_p=0.9,
        no_repeat_ngram_size=4)
    generated_text = list(map(tokenizer.decode, out))[0]
    print('---')
    print(generated_text)

