# Analysis of Box Office Trends Pre and Post-Pandemic: Examining Audience Attitudes Towards In-Theater Movie Releases

## Overview
This project analyzes audience attitudes towards box office trends before and after the pandemic. The analysis is based on a large sample of Reddit posts related to box office trends. The project uses clustering and zero-shot classification techniques to analyze the text data and gain insights into audience attitudes towards in-theater movie releases.

## Libraries
The code for this project is written in Python and uses the following libraries:

praw
pandas
numpy
sklearn
transformers

## Structure

The code is organized into several files:

- ```src/data.py```: This script contains the code for collecting the Reddit posts related to box office trends.

- ```src/bert_clustering.py```: This script contains the code to use the text data as BERT embeddings, and then clustering it using KMeans.

- ```src/bert_ngram.py```: This script contains the code to use the text data as BERT Ngram embeddings, and then clustering it using KMeans.

- ```src/Vector_ngram.py```: This script contains the code to use the text data as vectors and then using KMeans Clustering.

- ```src/zeroshot.py```: This script contains the code to perform zero shot classification using a pretrained contextualized language model such as BERT.

- ```data/``: This folder contains the data

- ```README.md```: This file provides an overview of the project and its code.

## Data
The data used in this project consists of Reddit posts related to box office trends. The data was collected using the Python Reddit API Wrapper (PRAW) and consists of over 10,000 posts from various subreddits. The data is stored in a CSV file and contains the text of the posts, as well as other metadata such as upvotes and comments.

## Results
The analysis of the Reddit posts revealed that the majority of the audience had mixed attitudes towards box office trends, with a slight negative sentiment towards post-pandemic releases. The use of clustering and zero-shot classification techniques enabled the analysis of a large amount of text data and provided insights into the attitudes of the audience towards in-theater movie releases.
 
 