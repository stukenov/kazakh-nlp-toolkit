import json
import os
from openai import OpenAI

PROMPT_KZ = (
    "Бұл аудио, жаңалықтар арнасының транскрипциясынан алынған мәтін, қазақ тілінде. "
    "Сіздің міндетіңіз маған осы мәтіннің грамматикалық түзетілген нұсқасын жіберу. "
    "Тек мәтінді жібер, басқа сөздерді жазба, уақыт кодтарын міндетті түрде қалдырыңыз. "
    "Жауабын JSON форматта жіберіңіз, өрістер тізімі қазіргі мәтіннен түзетілген нұсқасымен бірдей болады."
)

PROMPT = (
    "Ваша задача - отправить мне грамматически исправленную версию этого текста. "
    "Отправьте только текст, не пишите другие слова, обязательно оставьте временные коды. "
    "Ответ отправьте в формате JSON, список полей будет соответствовать исправленной версии текущего текста."
)

def get_openai_response(text, api_key=None):
    """
    Get OpenAI response for grammar correction.

    Args:
        text: Text to correct
        api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)

    Returns:
        Corrected text from OpenAI
    """
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": text}
        ]
    )
    return completion.choices[0].message.content

def process_json_file(file_path, api_key=None):
    """
    Process a JSON file with grammar correction.

    Args:
        file_path: Path to the input JSON file
        api_key: OpenAI API key (optional, will use env var if not provided)
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        json_content = json.load(file)

        i = 0
        while i < len(json_content):
            batch = []
            while i < len(json_content) and not json_content[i]['text'].strip().endswith(('.','!','?')):
                batch.append(json_content[i])
                i += 1
            if i < len(json_content):  # Add the sentence with the punctuation if it's not the end of the file
                batch.append(json_content[i])
                i += 1

            # Convert the batch into a JSON string before sending it to OpenAI
            batch_json = json.dumps(batch, ensure_ascii=False)
            corrected_batch_json = get_openai_response(batch_json, api_key)

            # Parse the corrected JSON content and update the original json_content
            corrected_batch = json.loads(corrected_batch_json)
            for j, corrected_subtitle in enumerate(corrected_batch):
                batch[j]['text'] = corrected_subtitle['text']

    grammar_file_path = f"content/captions/grammar/{file_path.split('/')[-1]}"
    with open(grammar_file_path, 'w', encoding='utf-8') as grammar_file:
        json.dump(json_content, grammar_file, ensure_ascii=False, indent=4)
    print(f"Grammar saved to {grammar_file_path}")

if __name__ == "__main__":
    # Example usage
    specific_json_file = "content/captions/json/eEf0BPwrdD8.json"
    process_json_file(specific_json_file)
