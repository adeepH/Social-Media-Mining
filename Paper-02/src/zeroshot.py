import pandas as pd
from transformers import pipeline
import sentencepiece
import numpy as np
import matplotlib.pyplot as plt

def zeroshot_classification(df, model_name):

  classifier = pipeline("zero-shot-classification", model=model_name)
  
  target_labels = ["better", "worse", "neutral"]

  sentences = list(df['text'])
  labels = []

  for sentence in sentences:
    result = classifier(sentence, target_labels)
    probs = result['scores']
    #append the label with maximum value
    labels.append(list(result['labels'])[0])
  
  return pd.DataFrame({
      'Comment': sentences,
      'labels': labels
  })

if __name__ == "__main__":
    df = pd.read_csv('clustered_data.csv')
    final_df = zeroshot_classification(df, 'finiteautomata/bertweet-base-sentiment-analysis')