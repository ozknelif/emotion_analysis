import pandas as pd


tweet_data = pd.read_csv('dataset\\train_text.txt', sep="\n", header=None, )
tweet_data.columns = ["tweets"]
print(tweet_data)
labels_data = pd.read_csv('dataset\\train_labels.txt', sep="\n", header=None, )
labels_data.columns = ["labels"]
labels = labels_data["labels"]
tweet_data = tweet_data.join(labels)
print(tweet_data)


def remove_unnecessary_columns(df, columns):
    for col in columns:
        del df[col]
    return df


df = remove_unnecessary_columns(df, ['Id', 'Date', 'Query', 'Author'])

pd.set_option('display.max_colwidth', None)

import re
def cleanTxt(text):
    text = re.sub('@[A-Za-z0–9_]+', '', text) #Removing @mentions
    text = re.sub('#', '', text) # Removing '#' hash tag
    text = re.sub('RT[\s]+', '', text) # Removing RT
    text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
    text = re.sub('-', '', text)
    text = text.lower()
    return text

df['Tweet'] = df['Tweet'].apply(cleanTxt)

print(df.iloc[0:10])
'''
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

allWords = ' '.join([twts for twts in df['Tweet']])
wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allWords)


plt.imshow(wordCloud, interpolation="bilinear")
plt.axis('off')
plt.show()
'''
'''
from textblob import TextBlob
i=0
for i in range(0 , len(df['Tweet'])):
  analysis = TextBlob(df['Tweet'][i])
  print(analysis.sentiment)
'''
'''
    from sklearn.feature_extraction.text import CountVectorizer
document = ["This is Import Data's YouTube channel",
            "Data Science is my passion and it is fun",
            "Please subscribe to my channel"]
# create the transform
vectorizer = CountVectorizer()

# tokenize and make the document into a matrix
doc_term_matrix = vectorizer.fit_transform(document)

pd.DataFrame(doc_term_matrix.toarray(),columns = vectorizer.get_feature_names())
'''
from sklearn.utils import shuffle
df = shuffle(df)
df.reset_index(inplace=True, drop=True)
processed_features = df.iloc[:10000, 1].values
labels = df.iloc[:10000, 0].values
print(processed_features)
print(labels)

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=1,
                                                    shuffle=True)
from sklearn.ensemble import RandomForestClassifier

text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)

predictions = text_classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))