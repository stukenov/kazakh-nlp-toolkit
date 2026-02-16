from datasets import load_dataset, Features, Value
import pandas as pd

# Define the features of the dataset for more control over its format
features = Features({
    'translation': {
        'kk': Value('string'),
        'en': Value('string')
    }
})
dataset = load_dataset("wmt19", "kk-en", split='train', features=features)

# Prepare the data for AutoTrain by creating prompts similar to Alpaca
data_for_autotrain = pd.DataFrame([{
    "text": f"<s>[INST] Translate Kazakh to English: {item['translation']['kk']} [/INST] {item['translation']['en']}</s>",
} for item in dataset])

# Ensure the 'data' directory exists
import os
if not os.path.exists('data'):
    os.makedirs('data')

# Save the prepared data in CSV format for AutoTrain in the 'data' directory
data_for_autotrain.to_csv("data/kazakh_to_english_instruct.csv", index=False, header=True, quoting=1)

prompt = "Write about USA"

prompt = f"[INSTRUCT]Translate Kazakh to English: {prompt.strip()}[/INSTRUCT]"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
attention_mask = inputs['attention_mask']
generate_ids = model.generate(inputs.input_ids, attention_mask=attention_mask, pad_token_id=model.config.eos_token_id, do_sample=True)
response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
print(response)

prompt = "<s>Write in Kazakh a detailed analogy between mathematics and a lighthouse.</s>"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
attention_mask = inputs['attention_mask']
generate_ids = model.generate(inputs.input_ids, attention_mask=attention_mask, pad_token_id=model.config.eos_token_id, max_length=128, do_sample=True)
response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
print(response)

