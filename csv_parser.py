import pandas as pd
from tqdm import tqdm
"""
Reads the csv dataset available at
http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip
and splits it into two files (.pos and .neg) containing the positive and
negative tweets.
Does some word preprocessing during the parsing.
"""


try:
    df = pd.read_csv('twitter-sentiment-dataset/sentiment-dataset.csv',
                     error_bad_lines=False)
    pos_dataset = open("twitter-sentiment-dataset/tw-data.pos", "w")
    neg_dataset = open("twitter-sentiment-dataset/tw-data.neg", "w")
except IOError:
    print("Failed to open file")
    quit()


df.SentimentText = df.SentimentText.str.strip()
df.SentimentText = df.SentimentText.str.replace(r'@.*', '<NAME/>')
df.SentimentText = df.SentimentText.str.replace(r'http://.*', '<LINK/>')
df.SentimentText = df.SentimentText.str.replace('#', '<HASHTAG/> ')
df.SentimentText = df.SentimentText.str.replace('&quot;', ' \" ')
df.SentimentText = df.SentimentText.str.replace('&amp;', ' & ')
df.SentimentText = df.SentimentText.str.replace('&gt;', ' > ')
df.SentimentText = df.SentimentText.str.replace('&lt;', ' < ')
df.SentimentText = df.SentimentText.str.replace(r'\W+', ' ')
df.SentimentText = df.SentimentText.str.strip()


for row in tqdm(df.itertuples(index=True, name='Pandas')):
    if row[2] == 1:
        pos_dataset.write(row[4] + "\n")
    else:
        neg_dataset.write(row[4] + "\n")
