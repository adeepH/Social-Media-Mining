# Importing the necessary libraries
import praw
import pandas as pd

# calling the reddit API 

reddit = praw.Reddit(client_id='5UhwM0Oh7-03hUDid4arew', # 
                     client_secret='pNIIdNz-2C07tI-d0uPpFxCqKX7BKQ',
                     user_agent='Kindly-Gate3544')


# Get the subreddit
subreddit_name = 'BoxOffice' 
keywords = ['pre-COVID', 'bomb', 'loss', 'post pandemic', 'future', 'uncertain', 
            'netflix', 'streaming', 'post-COVID', 'pandemic era', 'genre']

# Create an empty list to store the posts
post_list = []
# Iterate through the top 1000 posts in the subreddit
# Loop through the list of keywords and retrieve the top 1000 posts for each keyword
for keyword in keywords:
    subreddit = reddit.subreddit('BoxOffice')
    posts = subreddit.search(keyword, limit=1000)
    for post in posts:
        post_list.append([post.title, post.score, post.num_comments, post.url])

# Convert the list to a Pandas DataFrame
df = pd.DataFrame(post_list, columns=['text', 'upvotes', 'num_comments', 'url',])

# Remove duplicates based on post title
df.drop_duplicates(subset='text', keep='first', inplace=True)

# Sorting the dataframe based on the comments
#df_sorted = df.sort_values(by='upvotes', ascending=False)
# Export the data to a CSV file
df.to_csv('reddit_posts.csv', index=False)