from transformers import pipeline
classifier = pipeline('text-classification', model='gpt2')
result = classifier("Hello, I'm a language model,")
print(result)

