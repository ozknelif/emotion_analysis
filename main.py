import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# from nltk.stem import PorterStemmer
# from nltk.stem import WordNetLemmatizer


def read_data(file_path):
    colnames = ['Label', 'Id', 'Date', 'Query', 'Author', 'Tweet']
    df = pd.read_csv(file_path, encoding="ISO-8859-1", names=colnames, header=None)
    return df


def remove_unnecessary_columns(df, columns):
    for col in columns:
        del df[col]
    return df


def clean_tweet_data(text):
    text = re.sub('@[A-Za-z0–9_]+', '', text)  # Removing @mentions
    text = re.sub('#', '', text)  # Removing '#' hash tag
    text = re.sub('RT[\s]+', '', text)  # Removing RT
    text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink
    text = re.sub('-', '', text)
    text = text.lower()

    return text


def convert_data_to_token(data):
    tweet_tokens = data.apply(word_tokenize)
    return tweet_tokens


def main():
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # stop_words = set(stopwords.words('english'))

    tweet_data = read_data('../tweet_data.csv')
    tweet_data = remove_unnecessary_columns(tweet_data, ['Id', 'Date', 'Query', 'Author'])
    pd.set_option('display.max_colwidth', None)
    # print(tweet_data.iloc[0:10])
    tweet_data['Tweet'] = tweet_data['Tweet'].apply(clean_tweet_data)
    tweet_tokens = convert_data_to_token(tweet_data.Tweet)

main()
