from transformers import pipeline

translator = pipeline('translation', model="google/umt5-small")

text = "A <extra_id_0> walks into a bar and orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>."
translation = translator(text)
print(translation[0]['translation_text'])
