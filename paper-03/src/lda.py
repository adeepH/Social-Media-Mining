import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# load dataset
df = pd.read_csv('data/data.csv')

# create document-term matrix
vectorizer = CountVectorizer(max_features=1000, lowercase=False, ngram_range=(1,2))
dtm = vectorizer.fit_transform(df['text'])

# split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(dtm, df['sexist'], test_size=0.2, random_state=42)

# apply LDA on train set
lda = LatentDirichletAllocation(n_components=10, random_state=0)
lda_train = lda.fit_transform(X_train)

# train binary classification model

lr = LogisticRegression(random_state=0)
lr.fit(lda_train, y_train)


# infer on test set
lda_test = lda.transform(X_test)
y_pred = lr.predict(lda_test)

# calculate accuracy on test set
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set using LDA: {acc}")
