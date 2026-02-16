import fasttext
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

pretrained_lang_model = "lid218e.bin" # path of pretrained model file
model = fasttext.load_model(pretrained_lang_model)

text = "В Астану для участия в 10-м саммите Организации тюркских государств прибыл Президент Турции Реджеп Тайип Эрдоган, передает агентство Kazinform со ссылкой на пресс-службу правительства РК."

predictions = model.predict(text, k=1) 
input_lang = predictions[0][0].replace('__label__', '')

checkpoint = 'facebook/nllb-200-distilled-600M'
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

target_lang = 'kaz_Cyrl' # kaz_Cyrl 
translation_pipeline = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=input_lang, tgt_lang=target_lang, max_length=400)
output = translation_pipeline(text)
print(output[0]['translation_text'])
