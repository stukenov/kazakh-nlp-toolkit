import os
import json
from openai import OpenAI

PROMPT = (
    "Your task is to provide a grammatically corrected Kazakh translation of the given text. "
    "Please submit only the text, without any additional words, and ensure that time codes are retained. "
    "Send your response in JSON format, with a field list that matches the corrected version of the current text."
)

# Set your OpenAI API key as an environment variable: OPENAI_API_KEY
API_KEY = os.getenv("OPENAI_API_KEY")

def get_openai_translation_response(text, api_key):
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": text}
        ]
    )
    return completion.choices[0].message.content
    
def translate_to_kz_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        json_content = json.load(file)
        
        i = 0
        while i < len(json_content):
            batch = []
            while i < len(json_content) and not json_content[i]['text'].strip().endswith(('.','!','?')):
                batch.append(json_content[i])
                i += 1
            if i < len(json_content): 
                batch.append(json_content[i])
                i += 1
            
            batch_json = json.dumps(batch, ensure_ascii=False)
            translated_batch_json = get_openai_translation_response(batch_json, API_KEY)
            
            # Parse the translated JSON content and update the original json_content
            translated_batch = json.loads(translated_batch_json)
            for j, translated_subtitle in enumerate(translated_batch):
                batch[j]['text'] = translated_subtitle['text']
    translation_file_path = f"content/captions/translate_to_kz/{file_path.split('/')[-1]}"
    with open(translation_file_path, 'w', encoding='utf-8') as translation_file:
        json.dump(json_content, translation_file, ensure_ascii=False, indent=4)
    print(f"Translation saved to {translation_file_path}")

specific_json_file = "content/captions/translate_to_en/eEf0BPwrdD8.json"
translate_to_kz_json_file(specific_json_file)
