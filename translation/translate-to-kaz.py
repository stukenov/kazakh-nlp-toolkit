from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

article_en = "2) civil servant - a person who, in the manner prescribed by the legislation of the Republic of Kazakhstan, holds a paid staff position in state enterprises and government agencies and performs official duties in order to implement their tasks and functions, maintain and ensure functioning of state bodies;"
article_ru = "1) гражданская служба – профессиональная деятельность гражданских служащих по исполнению должностных полномочий, направленная на реализацию задач и функций казенных предприятий, государственных учреждений, осуществление технического обслуживания и обеспечение функционирования государственных органов;"

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# # translate English to Kazakh
# tokenizer.src_lang = "en_XX"
# encoded_en = tokenizer(article_en, return_tensors="pt")
# generated_tokens = model.generate(
#     **encoded_en,
#     forced_bos_token_id=tokenizer.lang_code_to_id["kk_KZ"]
# )
# translation_en_to_kk = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# print(translation_en_to_kk)

# translate Russian to Kazakh
tokenizer.src_lang = "ru_RU"
encoded_ru = tokenizer(article_ru, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_ru,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
)
translation_ru_to_en = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(translation_ru_to_en)
