from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import transformers
from nltk import ngrams
import torch 
# Load BERT tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
model = transformers.AutoModel.from_pretrained('bert-base-uncased')

# Function to generate n-gram embeddings using BERT
def generate_ngram_embeddings(text, n):
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    ngrams_list = list(ngrams(tokens, n))
    embeddings_list = []
    for ngram in ngrams_list:
        input_ids = tokenizer.encode(' '.join(ngram), add_special_tokens=True)
        input_ids = np.array(input_ids)
        input_ids = np.reshape(input_ids, (1, -1))
        outputs = model(torch.tensor(input_ids))[1].detach().numpy()
        embeddings_list.append(outputs)
    embeddings = np.mean(embeddings_list, axis=0)
    return embeddings

# Load data into Pandas dataframe
df = pd.read_csv('data.csv')

# Generate n-gram embeddings using BERT
ngram_size = 2
encoded_data = []
for text in df['text']:
    encoded_text = generate_ngram_embeddings(text, ngram_size)
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

# Generate word clouds for each cluster
for i in range(n_clusters):
    cluster_df = df[df['cluster'] == i]
    cluster_text = ' '.join(cluster_df['text'])
    wordcloud = WordCloud(background_color='white').generate(cluster_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Cluster {i}')
    plt.show()
