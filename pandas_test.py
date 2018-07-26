import pandas as pd

df = pd.read_csv('twitter-sentiment-dataset/sentiment-dataset.csv',
                  error_bad_lines=False)

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

df.to_csv('output.csv')

for row in df.itertuples(index=True, name='Pandas'):
        print(row[4])
