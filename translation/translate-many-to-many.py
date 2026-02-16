
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

def translate_text(text, model_name, source_lang, target_lang):
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    tokenizer.src_lang = source_lang
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translation

english_text = "He says his nightmare is being named and attacked by Beijing-backed media, which could cost him his job, or worse, his freedom. That fear has swept through Hong Kong's universities and academic circles, which once attracted top talent. The city was close to the mainland, yet far enough to host progressive classrooms, world-class libraries and archives that allowed academic freedom, even in Chinese studies."

translated_text = translate_text(english_text, "facebook/m2m100_1.2B", "en", "kk")
print(translated_text)

