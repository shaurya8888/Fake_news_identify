import streamlit as st
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer  # Import PorterStemmer from NLTK
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
news_df = pd.read_csv("C:/Users/Shaurya Singh/OneDrive/Documents/train.csv")
ps = PorterStemmer()

def stemming(content):
    if content is None:
        return "" 
# Return an empty string if content is None

    stemmed_content = re.sub('[^a-zA-Z]', " ", content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content

news_df['content'] = news_df['content'].apply(stemming)


news_df.head()
news_df.isna().sum()
news_df=news_df.fillna(' ')
news_df.isna().sum()
news_df['content']=news_df['author']+" "+news_df['title']
news_df

X=news_df['content'].values
Y=news_df['label'].values
vector=TfidfVectorizer()
vector.fit(X)
X=vector.transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,stratify=Y, random_state=1)
model=LogisticRegression()
model.fit(X_train,Y_train)
train_Y_pred=model.predict(X_train)
#print("train accuracy:",accuracy_score(train_Y_pred,Y_train))
test_Y_pred=model.predict(X_test)
#print("train accuracy:",accuracy_score(test_Y_pred,Y_test))
input_data=X_test[20:]
prediction=model.predict(input_data)
#if prediction[0]==1:
    #print("It is Fake news")
#else:
   # print("It is Real news")

st.title('FAKE NEWS DETECTOR')
input_text=st.text_input('Enter news article')

def prediction(input_text):
    input_data=vector.transform([input_text])
    prediction=model.predict(input_data)
    return prediction[0]
if input_text:
    pred=prediction(input_text)
    if pred==1:
        st.write("The news is Fake")
    else:
        st.write("The news is Real")

    