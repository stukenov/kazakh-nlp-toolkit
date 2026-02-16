from fastai.text.all import load_learner
from fastapi import FastAPI

model = load_learner('model/model.pkl')

app = FastAPI()

@app.get("/classify/{text}")
async def classify_text(text: str):
    prediction = model.predict(text)
    category = 'pos' if prediction[2][1] > prediction[2][0] else 'neg'
    confidence = round(max(prediction[2]).item() * 100, 2)
    return {"category": category, "confidence": f"{confidence}%"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
