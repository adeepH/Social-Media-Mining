import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Load the data
df = pd.read_csv('reddit_posts.csv', encoding='utf-8')

# Remove NaN values
df.dropna(inplace=True)

# Preprocess the text data
df['text'] = df['text'].str.lower()
# Create the feature vectors
vectorizer = CountVectorizer(ngram_range=(2, 5))
X = vectorizer.fit_transform(df['text'])

# Cluster the data
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
df['cluster'] = kmeans.labels_

# Print the top keywords for each cluster
# Get the feature names
feature_names = list(vectorizer.vocabulary_.keys())
feature_names.sort(key=lambda x: vectorizer.vocabulary_[x])
#terms = vectorizer.get_feature_names()
for i in range(5):
    print('Cluster {}:'.format(i))
    keywords = [feature_names[index] for index in kmeans.cluster_centers_.argsort()[:,::-1][i,:10]]
    print(keywords)
# Cluster the data
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
df['cluster'] = kmeans.labels_
# Generate the wordclouds
fig, axs = plt.subplots(1, 5, figsize=(12, 4))
   
for i, ax in enumerate(axs.flatten()):
    text = " ".join(df[df['cluster'] == i]['text'])
    wordcloud = WordCloud(width=800, height=800, background_color='white', collocations=False).generate(text)

    # Add the WordCloud to the subplot
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Cluster {i+1}')

# Display the subplots
df['cluster'] = df['cluster'].astype(int)
print(df['cluster'].isna().sum())
df2 = df[(df['cluster'] == 2) | (df['cluster'] == 4) | (df['cluster'] == 5)]
df2.to_csv('clustered_data.csv', index=False)
plt.tight_layout()
plt.show()