import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# load the dataset
data = pd.read_csv('data/data.csv', header=None)

# vectorize tweet text using a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data[0])


# perform NMF-based topic modeling
num_topics = 10
nmf_model = NMF(n_components=num_topics, random_state=42)
nmf_model.fit(X)

# print top words for each topic
feature_names = vectorizer.get_feature_names()
for topic_idx, topic in enumerate(nmf_model.components_):
    print(f"Topic #{topic_idx}:")
    print(", ".join([feature_names[i] for i in topic.argsort()[:-6:-1]]))
    print()
