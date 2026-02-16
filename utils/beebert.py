from transformers import pipeline

unmasker = pipeline('fill-mask', model='kz-transformers/kaz-roberta-conversational')
result = unmasker("Сәлем, мен <mask> қаласында тұрамын.")
print(result)
