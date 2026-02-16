from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "RWKV/rwkv-raven-1b5"

model = AutoModelForCausalLM.from_pretrained(model_id).to(0)
tokenizer = AutoTokenizer.from_pretrained(model_id)

question = "Tell me about ravens"
prompt = f"### Instruction: {question}\n### Response:"

inputs = tokenizer(prompt, return_tensors="pt").to(0)
output = model.generate(inputs["input_ids"], max_new_tokens=100)

print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))