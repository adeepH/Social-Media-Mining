import pandas as pd
import numpy as np
import re
import nltk
import gensim
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import HdpModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix

def preprocess_tweet_text(tweet):
    # remove URLs and mentions
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    
    # remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    
    # convert to lower case
    tweet = tweet.lower()
    
    # tokenize tweet text
    tokens = nltk.word_tokenize(tweet)
    
    # remove stop words
    tokens = [token for token in tokens if token not in stop_words]
    
    return tokens

# load dataset
df = pd.read_csv('data/data.csv')

# create a dictionary and a corpus
# create dictionary

# create document-term matrix
vectorizer = CountVectorizer(max_features=1000, lowercase=False, ngram_range=(1,2))
dtm = vectorizer.fit_transform(df['text'])

# split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(dtm, df['sexist'], test_size=0.2, random_state=42)

# train HDP model on training data
hdp_model = gensim.models.hdpmodel.HdpModel(corpus=gensim.matutils.Sparse2Corpus(X_train),
                                            id2word=dict((v, k) for k, v in vectorizer.vocabulary_.items()))

# infer topic distribution for test data
test_topic_dist = hdp_model.inference(gensim.matutils.Sparse2Corpus(X_test))

# fit logistic regression model on training data
lr = LogisticRegression()
lr.fit(test_topic_dist[0], y_train)

# evaluate accuracy on test data
y_pred = lr.predict(test_topic_dist[1])
accuracy = np.mean(y_pred == y_test)
print("Test accuracy:", accuracy)