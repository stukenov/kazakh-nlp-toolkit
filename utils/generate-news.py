import os
import argparse
from openai import OpenAI
import json
import uuid

# Set your OpenAI API key as an environment variable: OPENAI_API_KEY
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_news(title):

    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": f"Generate a article about {title}."},
      ])

    unique_id = str(uuid.uuid4())
    with open(f'{unique_id}.txt', 'w') as f:
        f.write(response['choices'][0]['message']['content'])
    print(f'{unique_id}.txt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate news using OpenAI GPT-4')
    parser.add_argument('title', type=str, help='Title of the news theme')

    args = parser.parse_args()

    generate_news(args.title)
