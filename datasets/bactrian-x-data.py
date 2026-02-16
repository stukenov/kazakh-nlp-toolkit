import json
from datasets import load_dataset, DatasetDict, DatasetInfo, SplitGenerator, Value, Features, Version, BuilderConfig, GeneratorBasedBuilder

LANGS = ['kk', 'ru', 'en']

_URL = "https://huggingface.co/datasets/MBZUAI/Bactrian-X/resolve/main/data/"
_URLS = {lang: _URL + f"{lang}.json.gz" for lang in LANGS}

class BactrianConfig(BuilderConfig):
    """BuilderConfig for Bactrian."""

    def __init__(self, lang, **kwargs):
        """BuilderConfig for Bactrian."""
        super().__init__(version=Version("1.0.1"), **kwargs)
        self.lang = lang
        self.data_url = _URLS[lang]

class Bactrian(GeneratorBasedBuilder):
    """The Bactrian Datasets."""

    BUILDER_CONFIGS = [BactrianConfig(name=lang, description=f"Bactrian dataset for the {lang} language.", lang=lang) for lang in LANGS]

    def _info(self):
        return DatasetInfo(
            description='Bactrian-X',
            features=Features({
                'instruction': Value('string'),
                'input': Value('string'),
                'id': Value('string'),
                'output': Value('string'),
            }),
            homepage='',
            citation='',
        )

    def _split_generators(self, dl_manager):
        downloaded_file = dl_manager.download_and_extract(self.config.data_url)
        return [
            SplitGenerator(name='train', gen_kwargs={"data_file": downloaded_file}),
        ]

    def _generate_examples(self, data_file):
        with open(data_file, 'r', encoding="utf-8") as f:
            for id, instance in enumerate(json.load(f)):
                yield id, {
                    "instruction": instance['instruction'],
                    "input": instance['input'],
                    "output": instance['output'],
                    "id": instance['id']
                }

print('Loading dataset...')
dataset = load_dataset('77-bactrian-x-data.py', 'kk')
print('Loaded dataset:', len(dataset['train']))
print('Example:', dataset['train'][0])
print('Example:', dataset['train'][10])