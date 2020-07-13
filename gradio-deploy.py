import gradio as gr
from src import prediction

gr.Interface(fn=prediction.predict_sentiment, 
            inputs = gr.inputs.Textbox(lines=7, label="Review Text"),
            # inputs=["textbox"], 
            outputs=gr.outputs.Textbox(label="Review Sentiment")).launch()