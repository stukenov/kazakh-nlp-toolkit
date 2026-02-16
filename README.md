# Kazakh NLP Toolkit

A comprehensive collection of Natural Language Processing (NLP) tools and scripts for the Kazakh language. This toolkit includes utilities for translation, model training, dataset preparation, tokenization, and more.

## Features

- **Translation**: Multiple translation tools for Kazakh ↔ English/Russian
- **Model Training**: Scripts for fine-tuning GPT-2, mGPT, RWKV, and other language models
- **Dataset Preparation**: Tools for processing and preparing Kazakh language datasets
- **Utilities**: Tokenization, summarization, text generation, and more

## Project Structure

```
kazakh-nlp-toolkit/
├── translation/      # Translation scripts
├── training/         # Model training and fine-tuning
├── datasets/         # Dataset preparation and processing
└── utils/           # Utility scripts
```

## Translation Scripts

### `/translation`

- **translate.py** - M2M100-based translation (Spanish to Russian)
- **translation.py** - Basic translation utility
- **translate-article.py** - Article translation using OpenAI GPT
- **translate-many-to-many.py** - Multi-language translation support
- **translate-to-kaz.py** - Translation to Kazakh
- **translate_to_kazakh.py** - JSON-based translation to Kazakh with GPT-4
- **translate_to_english.py** - JSON-based translation to English with GPT-4
- **translate2.py** - Batch translation with concurrent processing
- **translated-gpt.py** - GPT-powered translation

## Training Scripts

### `/training`

- **gpt2.py** - Basic GPT-2 model usage
- **gpt2_espanol.py** - Comprehensive GPT-2 training for conversational AI (Spanish example)
- **mgpt_tuning_kazakh.py** - Fine-tuning mGPT-1.3B on Kazakh text (Abay Joly dataset)
- **mgpt_tuning_example.py** - Example mGPT fine-tuning script
- **rugpt-qaz-test.py** - Testing ruGPT with Kazakh text
- **rugpt35_13b.py** - ruGPT-3.5 13B model implementation
- **rwkv.py** - RWKV model implementation
- **rwkw4.py** - RWKV-4 model usage
- **train-llm.py** - Generic LLM training script
- **train-llm2.py** - Alternative LLM training implementation
- **train-llm-instruct.py** - Instruction-based LLM training
- **finetune-llm.py** - General fine-tuning utilities
- **falcon_guanaco_ipynb_.py** - Falcon model with Guanaco dataset
- **Xwin-LM.py** - Xwin language model implementation

## Dataset Preparation

### `/datasets`

- **prepare-dataset-for-nanogpt.py** - Dataset preparation for nanoGPT with extensive data processing utilities
- **prepare-text.py** - Text preprocessing and cleaning for Kazakh language
- **dataset_rulm.py** - RULM dataset preparation
- **bactrian-x-data.py** - Bactrian-X dataset processing
- **get-kaz-content.py** - Kazakh content extraction
- **undup.py** - Deduplication utility for datasets

## Utility Scripts

### `/utils`

- **tokenize.py** - BPE tokenizer training and tokenization for Kazakh
- **summarize-every-links.py** - News article summarization using GPT-4
- **generate-news.py** - News article generation using GPT
- **test-nltk-in-kazakh.py** - NLTK testing for Kazakh language
- **test-bart-classification.py** - BART model classification testing
- **torch-rnn-work.py** - RNN-based text generation
- **fastai-fastapi.py** - FastAI and FastAPI integration
- **tinystories.py** - TinyStories dataset utilities
- **beebert.py** - BERT model utilities
- **umt5-small.py** - UMT5 small model implementation

## Installation

```bash
# Clone the repository
git clone https://github.com/stukenov/kazakh-nlp-toolkit.git
cd kazakh-nlp-toolkit

# Install dependencies (example for common requirements)
pip install transformers torch openai nltk tokenizers tqdm pandas numpy
```

## Configuration

Some scripts require API keys for OpenAI services. Set your API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage Examples

### Translation

```python
# Translate text to Kazakh using M2M100
python translation/translate.py

# Translate JSON subtitles to Kazakh
python translation/translate_to_kazakh.py
```

### Training

```python
# Fine-tune mGPT on Kazakh text
python training/mgpt_tuning_kazakh.py

# Train GPT-2 model
python training/gpt2_espanol.py
```

### Dataset Preparation

```python
# Prepare dataset for nanoGPT
python datasets/prepare-dataset-for-nanogpt.py

# Clean and prepare Kazakh text
python datasets/prepare-text.py
```

### Utilities

```python
# Train BPE tokenizer
python utils/tokenize.py

# Summarize news articles
python utils/summarize-every-links.py
```

## Models Used

This toolkit works with various pre-trained models:

- **GPT-2**: General text generation
- **mGPT-1.3B**: Multilingual GPT including Kazakh support
- **ruGPT-3.5**: Russian GPT models
- **RWKV**: Efficient RNN-based language models
- **M2M100**: Multilingual translation
- **BART**: Classification and NLU tasks
- **Falcon**: Instruction-following models

## Datasets

The toolkit supports various Kazakh language datasets:

- Multidomain Kazakh Dataset
- Abay Joly (Kazakh literature)
- RULM dataset
- Bactrian-X multilingual dataset
- Custom Kazakh news and books datasets

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Saken Tukenov** ([@stukenov](https://github.com/stukenov))

## Acknowledgments

- Hugging Face Transformers
- OpenAI
- AI Forever (mGPT)
- KZ-Transformers team
- All open-source contributors to the Kazakh NLP community

## Notes

- Some scripts are adapted from Colab notebooks and may require modifications for local execution
- GPU is recommended for training scripts
- Not all scripts have been tested in production environments
- API keys are required for scripts using OpenAI services

---

**Disclaimer**: This is a collection of experimental and research scripts. Use at your own discretion.
