from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Example corpus of documents
df = pd.read_csv('data.csv')
print(df.shape)
posts = df['Title']
# Get the keywords associated with each cluster
# Create a vectorizer to convert text to a matrix of word n-gram features
vectorizer = TfidfVectorizer(ngram_range=(2,4))
X = vectorizer.fit_transform(posts)

# Use K-Means clustering to cluster the documents
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X)

# Get the keywords associated with each cluster
feature_names = vectorizer.get_feature_names_out()
keywords_per_cluster = []
cluster_center = kmeans.cluster_centers_

for i in range(num_clusters):
    top_indices = np.argsort(cluster_center)[::-1][:10]  # top 10 features
    keywords = [feature_names[j] for j in top_indices]
    keywords_per_cluster.append(keywords)

# Print the keywords associated with each cluster
for i in range(num_clusters):
    #print(f"Cluster {i+1} keywords: {', '.join(keywords_per_cluster[i])}")  
    keywords = [keywords_per_cluster[ind] for ind in cluster_center[i, :].argsort()[::-1][:10]]
    wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(' '.join(keywords))
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title(f'Cluster {i+1}', fontsize=20)
    plt.show()

print(keywords_per_cluster)

# Elbow method
