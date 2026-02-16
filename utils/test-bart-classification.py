from transformers import pipeline

# This model is a `zero-shot-classification` model.
# It will classify text, except you are free to choose any label you might imagine
classifier = pipeline(model="facebook/bart-large-mnli")
show = classifier(
    "У меня проблема с моим iPhone, которую нужно решить как можно скорее!!",
    candidate_labels=["срочно", "не срочно", "телефон", "планшет", "компютер"],
)

print(show)
