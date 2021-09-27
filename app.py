
"""
@author: Gokulraj
"""
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from fuzzywuzzy import process
import numpy as np
import pickle
import pandas as pd
import webbrowser
import streamlit as st 

from PIL import Image
#cs=pickle.load(open('cosine.pkl', 'rb'))
df = pd.read_csv("movies_final.csv")

def top_10(movie_title):
    cv = CountVectorizer()
    cv = cv.fit_transform(df["importent_feature"])
    cs = cosine_similarity(cv)
    title = process.extractOne(movie_title,df["movie_title"])
    if title[1] > 80:
        movie_id = title[2]
        scores = list(enumerate(cs[movie_id]))
        sorted_score = sorted(scores,key = lambda x:x[1],reverse=True)
        top_10 = sorted_score[1:11]
        ans = []
        for i,j in zip(top_10,range(1,len(top_10)+1)):
            movie_title = df["movie_title"][i[0]]
            ans.append(movie_title)
            
            
        return ans
    else:
        ans  = "Sorry! The movie you requested is not in our database. Please check the spelling or try with other movies!"
        return ans

def main():
    st.text("@author: Gokulraj.T")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Movies Recommendation engine</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    message = st.text_input("Enter the movie title")
    if st.button("Predict"):
        result=top_10(message)
        if type(result) == str:
            st.success(result)
        else:
            for i,j in zip(range(1,11),result):
                st.text(f"{i}.{j}")
   
    if st.button("About"):
        st.text("This app giving the recommentations based on the movie genres and cast&crew and title.") 
        st.text("Recommendations given by : Cosine similarity")
        link = '[Code](https://github.com/gokulvm/Movie_recommentation_system_based_on_cosine_similarity)'
        st.markdown(link, unsafe_allow_html=True)
       
if __name__=='__main__':
    main()
    
       
