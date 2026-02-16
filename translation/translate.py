import csv
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from torch.utils.data import DataLoader, Dataset

class TranslationDataset(Dataset):
    def __init__(self, read_file):
        self.data = []
        with open(read_file, mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.data.extend(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Initialize the tokenizer and model only once to improve performance
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
model = model.to('cuda')

def translate(text):
    tokenizer.src_lang = "es"
    encoded = tokenizer(text, return_tensors="pt").to('cuda')
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("ru"))
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

def process_translation(batch):
    return [translate(text) for text in batch]

# Create dataset
dataset = TranslationDataset('71-final_es_conv.csv')
dataloader = DataLoader(dataset, batch_size=32, collate_fn=lambda x: x)

for batch in dataloader:
    # Process each batch in parallel
    translated_batch = process_translation(batch)
    for translated_text in translated_batch:
        print(translated_text)

