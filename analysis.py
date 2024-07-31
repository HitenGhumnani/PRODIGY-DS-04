# Import necessary libraries
import pandas as pd
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Download necessary NLTK corpora
nltk.download('punkt')

# Load the data
file_path = 'data.csv'  # Update this path to your downloaded file location
data = pd.read_csv(file_path)

# Check the columns in the DataFrame
print(data.columns)

# Display the first few rows of the DataFrame
print(data.head())

# Assuming the correct column name is identified (here using the fourth column as an example)
text_column = data.columns[3]  # Adjust the index if needed based on inspection

# Function to clean the tweet text
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\S+', '', text)     # Remove mentions
        text = re.sub(r'#\S+', '', text)     # Remove hashtags
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
    else:
        text = ''
    return text

# Apply cleaning function to the text column
data['cleaned_text'] = data[text_column].apply(clean_text)

# Function to get sentiment
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Apply sentiment analysis function
data['sentiment'] = data['cleaned_text'].apply(get_sentiment)

# Plot sentiment distribution
sentiment_counts = data['sentiment'].value_counts()
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
plt.title('Sentiment Analysis of Tweets')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
