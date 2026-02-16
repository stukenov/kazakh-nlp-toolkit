from datasets import load_dataset, Features, Value
import pandas as pd

# Define the features of the dataset for more control over its format
features = Features({
    'translation': {
        'kk': Value('string'),
        'en': Value('string')
    },
    'INSTRUCTION': Value('string'),  # Added feature for instruction
    'RESULT': Value('string')        # Added feature for result
})
dataset = load_dataset("wmt19", "kk-en", split='train', features=features)

# Prepare the data for AutoTrain by creating prompts similar to Alpaca
data_for_autotrain = pd.DataFrame([{
    "text": f"Translate Kazakh to English: {item['translation']['kk']}\n{item['translation']['en']}",
    "INSTRUCTION": "Translate the following Kazakh text to English.",  # Added instruction
    "RESULT": item['translation']['en']  # Added result
} for item in dataset])

# Ensure the 'data' directory exists
import os
if not os.path.exists('data'):
    os.makedirs('data')

# Save the prepared data in CSV format for AutoTrain in the 'data' directory
data_for_autotrain.to_csv("data/kazakh_to_english.csv", index=False, header=True, quoting=1)
