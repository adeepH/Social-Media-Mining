import pandas as pd
from sklearn.model_selection import train_test_split

import re
import string
import nltk

# load stop words
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# load the dataset
data = pd.read_csv('data/data.csv', header=None)

# split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


def preprocess_tweet_text(tweet):
    # remove URLs and mentions
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    
    # remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    
    # convert to lower case
    #tweet = tweet.str.lower()
    
    # tokenize tweet text
    tokens = nltk.word_tokenize(tweet)
    
    # remove stop words
    tokens = [token for token in tokens if token not in stop_words]
    
    return tokens

# preprocess the training data
train_tweets = train_data[0].apply(preprocess_tweet_text).tolist()

# preprocess the testing data
test_tweets = test_data[0].apply(preprocess_tweet_text).tolist()

from sklearn.feature_extraction.text import TfidfVectorizer

# vectorize tweet text using a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_tweets)
X_test = vectorizer.transform(test_tweets)

from sklearn.decomposition import TruncatedSVD

# fit LSA model on training data
lsa_model = TruncatedSVD(n_components=50, random_state=42)
lsa_model.fit(X_train)

# transform training and testing data to topic space
X_train_lsa = lsa_model.transform(X_train)
X_test_lsa = lsa_model.transform(X_test)


from sklearn.linear_model import LogisticRegression

# train logistic regression model on transformed training data
lr = LogisticRegression()
lr.fit(X_train_lsa, train_data[1])

# evaluate accuracy on transformed test data
accuracy = lr.score(X_test_lsa, test_data[1])
print("Test accuracy:", accuracy)

