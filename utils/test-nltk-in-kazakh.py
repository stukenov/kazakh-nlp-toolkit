import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

sentence = """Сәлеметсіз бе? Бүгін сіз қалайсыз?"""

tokens = nltk.word_tokenize(sentence, language='kazakh')

print(tokens)