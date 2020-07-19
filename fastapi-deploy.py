## uvicorn fastapi-deploy:app --reload
from fastapi import FastAPI
from src import prediction

app = FastAPI()

@app.get('/')
def read_route():
    return {"Hello":"World  "}


@app.post("/reviews/")
def predict_sentiment(review_text):
    sentiment = 'empty_string'
    if len(review_text)>0:
        sentiment = prediction.predict_sentiment(review_text)
    return sentiment