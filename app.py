# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
import streamlit as st
import google.generativeai as genai
import os
import nltk
import pandas as pd
import PyPDF2 as pdf
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

st.set_page_config(
    page_title="Streamly - ATS NLP",
    page_icon="imgs/nlp.png"
)

nltk.download('punkt') 
nltk.download('stopwords')
stopwords_words = stopwords.words('english')
stopwords_words.extend(['and', 'as', 'at', 'in',"of","on", 'the','the','with'])

 

#load enviroment variabls from a .env file
load_dotenv()



#Extract info PDF


def input_pdf_text(uploaded_file):
	reader=PdfReader(uploaded_file)
	text = ""
	for page in reader.pages:
		text += str(page.extract_text())
	return text




##streamlit app
st.markdown("<h1 style='text-align: center; color: #9D4BFF;'>ATS with NLP</h1>", unsafe_allow_html=True)

#st.title("ATS NLP")
st.text("Compare your CV with jobs descriptions to finding the best macth and Unlock next job")

jd=st.text_area("Paste the job description")
uploaded_file=st.file_uploader("Upload Your Resume",type="pdf",help="please upload the pdf")
submit = st.button("Submit")

text1 = input_pdf_text(uploaded_file)

word_tokens = word_tokenize(text1)
filtered_sentence = [w for w in word_tokens if not w in stopwords_words]
#with no lower case conversion
filtered_sentence = []
 
for w in word_tokens:
    if w not in stopwords_words:
        filtered_sentence.append(w)

documents = [jd,text1]

tdidf_vectorize = TfidfVectorizer()
sparse_matrix =tdidf_vectorize.fit_transform(documents)

doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(
	doc_term_matrix,
	columns=tdidf_vectorize.get_feature_names_out(),
	index=["jd","uploaded_file"],
)
similarity_score= cosine_similarity(df,df)[0,1]
match_keys = df.isin([0]).sum(axis=0)
match_words = match_keys[match_keys.values == 0].keys()
match_words = list(match_words)
match_words = [w for w in match_words if not w in stopwords_words]
score=round(similarity_score,2)*100
score=int(score)
html_str = f"""
<style>
p.a {{
  text-align: center;	
  font: bolder 80px Courier;
  color: #9D4BFF;
}}
</style>
<p class="a">
{score}%</p>
"""



if submit:
	if uploaded_file is not None:
		#text1=input_pdf_text(uploaded_file)
		st.markdown(html_str, unsafe_allow_html=True)
		st.markdown("<p style='text-align: center; color: #ffffff;'>Analizamos la oferta y tu hoja de vida y tienes una compatibilidad</p>", unsafe_allow_html=True)
		s=''
		for i in match_words:
			s += "- " + i + "\n"
		st.markdown(s, unsafe_allow_html=True)
