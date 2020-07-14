# run: streamlit run streamlit-deploy.py 
import streamlit as st
from src import prediction



def main():

    st.title('Movie Reviews Classification')
    sentence = st.text_input('Input your review here:') 
    pred = prediction.predict_sentiment(sentence)

    if sentence:
        if pred == "positive":
            st.success(pred)
        elif pred == 'negative':
            st.error(pred)


if __name__ == '__main__':
    main()
