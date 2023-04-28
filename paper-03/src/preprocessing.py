import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split, GridSearchCV


# load dataset
fname = 'sexism_data.csv'
df = pd.read_csv(fname)

df = df[['text', 'sexist']]
df['sexist'] = df['sexist'].apply({
    False:0, True:1
}.get)

#df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

#df_train = pd.read_csv('train.csv')
#df_test = pd.read_csv('test.csv')


# preprocess text data
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text) # remove non-alphanumeric characters
    text = re.sub(r'\d+', '', text) # remove digits
    text = text.lower() # convert to lowercase
    words = text.split() # split into words
    stop_words = set(stopwords.words('english')) # get stopwords
    words = [w for w in words if not w in stop_words] # remove stopwords
    text = ' '.join(words) # join words back to sentence
    return text

#df_train['text'] = df_train['text'].apply(preprocess_text)
#df_test['text'] = df_test['text'].apply(preprocess_text)

df['text'] = df['text'].apply(preprocess_text)
path = 'data'
if not os.path.exists(path):
    os.mkdir(path)

df.to_csv(f'{path}/data.csv',index=False)
#df_train.to_csv(f'{path}/train.csv', index=False)
#df_test.to_csv(f'{path}/test.csv', index=False)
