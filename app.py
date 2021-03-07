import streamlit as st
from pyngrok import ngrok
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pandas as pd
st.title('Spam Ham Classification')
df = pd.read_table('spam.tsv')
x = df.iloc[:,1].values
y = df.iloc[:,0].values
text_model = Pipeline([('tfidf',TfidfVectorizer()),('model',SVC())])
text_model.fit(x,y)
select = st.text_input('Enter your message')
op = text_model.predict(select)
st.title(op)
