import gradio as gr
from prediction import predict_sentiment


print(predict_sentiment("hello"))

gr.Interface(fn=predict_sentiment, 
            inputs = gr.inputs.Textbox(lines=7, label="Review Text"),
            # inputs=["textbox"], 
            outputs=gr.outputs.Textbox(label="Review Sentiment")).launch()