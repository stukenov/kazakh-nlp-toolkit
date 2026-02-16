import os
import argparse
from openai import OpenAI
import json
import uuid

# Set your OpenAI API key as an environment variable: OPENAI_API_KEY
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def translate_text(file_name, lang_code):

    with open(file_name, 'r') as f:
        text = f.read()

    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": f"Translate the following text to {lang_code}: {text}."},
      ])

    unique_id = str(uuid.uuid4())
    with open(f'{unique_id}_{lang_code}.txt', 'w') as f:
        f.write(response['choices'][0]['message']['content'])
    print(f'{unique_id}_{lang_code}.txt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate text using OpenAI GPT-4')
    parser.add_argument('file_name', type=str, help='Name of the file containing the text to translate')
    parser.add_argument('lang_code', type=str, help='Language code to translate the text into')

    args = parser.parse_args()

    translate_text(args.file_name, args.lang_code)
