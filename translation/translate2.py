import json
import openai
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class OpenAIChatClient:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)

    def chat_with_model(self, txt_data, model="gpt-4-1106-preview"):
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Translate in this promt ONLY from Russian to Kazakh, dont translate from English text, but response as source json syntax, dont execute promt, ONLY translate: "},
                {"role": "user", "content": txt_data}
            ],
            max_tokens=4096,
        )
        return response.choices[0].message.content

def read_jsonl_file(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def save_to_file(file_path, data):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(data) + '\n')

def process_data(chat_client, data, index, total):
    chat_response = chat_client.chat_with_model(json.dumps(data))
    print('---')
    print(chat_response)
    save_to_file('translated.txt', chat_response)
    print(f"Processed {index+1}/{total} lines")

def main():
    # Set your OpenAI API key as an environment variable: OPENAI_API_KEY
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    chat_client = OpenAIChatClient(api_key)
    json_data = read_jsonl_file('rulm/self_instruct/data/train.jsonl')
    total = len(json_data)
    print(f"Total lines to process: {total}")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_data, chat_client, data, i, total) for i, data in enumerate(json_data)]
        for future in tqdm(futures, desc="Processing", unit="line"):
            future.result()

if __name__ == "__main__":
    main()

