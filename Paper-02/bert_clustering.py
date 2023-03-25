import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud 
from transformers import BertTokenizer, BertModel
import torch
import nltk
from nltk import ngrams
nltk.download('punkt')
from collections import Counter
#from keras.preprocessing.sequence import pad_sequences

# Preprocess your data as needed
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# load the dataset
df = pd.read_csv('reddit_posts.csv')
# Function to encode text using BERT
def encode_text(text):
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    input_ids = np.array(input_ids)
    input_ids = np.reshape(input_ids, (1, -1))
    outputs = model(torch.tensor(input_ids))[1].detach().numpy()
    return outputs

encoded_data = []
for text in df['text']:
    encoded_text = encode_text(text)
    encoded_data.append(encoded_text)
encoded_data = np.array(encoded_data)
# Reshape the 3D array into a 2D array
n_samples = encoded_data.shape[0]
n_features = encoded_data.shape[1] * encoded_data.shape[2]
encoded_data_2d = np.reshape(encoded_data, (n_samples, n_features))

# Apply KMeans clustering to the 2D array
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(encoded_data_2d)

# Add cluster labels to dataframe
df['cluster'] = kmeans.labels_

# Get keywords for each cluster
cluster_keywords = []
for i in range(n_clusters):
    cluster_df = df[df['cluster'] == i]
    cluster_text = ' '.join(cluster_df['text'])
    wordcloud = WordCloud(background_color='white').generate(cluster_text)
    keywords = list(wordcloud.words_.keys())
    cluster_keywords.append(keywords)

# Print the keywords for each cluster
#for i in range(n_clusters):
#    print(f'Cluster {i} keywords: {cluster_keywords[i]}')
    tokens = nltk.word_tokenize(cluster_text)
    ngram_counts = Counter()
    for n in range(2, 5):
        ngrams_list = list(ngrams(tokens, n))
        ngram_counts += Counter(ngrams_list)
        
    # Extract top 10 most frequent n-grams
    cluster_keywords = [x[0] for x in ngram_counts.most_common(10)]
    print(f'Cluster {i} keywords: {cluster_keywords}')

"""# Generate word clouds for each cluster
for i in range(n_clusters):
    cluster_df = df[df['cluster'] == i]
    cluster_text = ' '.join(cluster_df['text'])
    wordcloud = WordCloud(background_color='white').generate(cluster_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Cluster {i}')
    plt.show()
"""