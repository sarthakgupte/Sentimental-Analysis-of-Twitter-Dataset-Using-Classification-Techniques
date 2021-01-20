"""
Author: Sarthak Gupte

This code is written to perform sentimental analysis on the Twitter data set. To classify the data between positive and
negative tweet. To perform this here we perform data cleaning and prepation, and also used classification technique to
evaluate different models.
"""
import re
import string
#import tweepy
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import porter
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def drop_columns(columns, data):
    """
    Column which are of no use for classification are dropped and update data frame is returned.
    :param columns: Columns to be deleted
    :param data: Data frame
    :return: Updated Data Frame
    """
    for column in columns:
        del data[column]
    return data


# def tweepy_api():
#     """
#     This function is shown as an example, this was used to scrap the data inititally. Since the secret keys of your
#     twitter account is generated hence, it is not used any further. Moving on we used pre-labeled data for this project.
#     :return:
#     """
#     auth = tweepy.OAuthHandler("CONSUMER_KEY", "CONSUMER_SECRET")
#     auth.set_access_token("ACCESS_TOKEN", "ACCESS_TOKEN_SECRET")
#     api = tweepy.API(auth)
#
#     tweets = api.user_timeline()
#     for tweet in tweets:
#         print(tweet.text)
#
#     try:
#         api.verify_credentials()
#         print("Authentication OK")
#     except:
#         print("Error during authentication")


def word_cloud(data):
    """
    This function is used to display the words in a picture. Bigger the word more is the frequency of the words in
    different tweets.
    :param data: tweets
    :return: None
    """
    text = ' '.join([word for word in data])
    cloud = WordCloud(width=600, height=600, background_color='black').generate(text)
    # plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(cloud)
    # plt.axis("off")
    # plt.tight_layout(pad=0)

    plt.show()


def cleaning(text):
    """
    This function is used to pre process the data for better results. We remove stopwords, punctuation, numbers, URL,
    words with length less than 3 and converted each word to their respective roots.
    :param text: Single tweet
    :return: None
    """
    # print(text)
    stop_words = stopwords.words("english")
    stemming = porter.PorterStemmer()

    text = re.sub(r'\@\w+', '', text)

    text = re.sub(r'[0-9]+', '', text)

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = re.sub(r'\bhttp\w+', '', text)

    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([stemming.stem(word) for word in text.split()])
    text = ' '.join([word for word in text.split() if len(word) > 2])
    # text = [word for word in text if len(word) > 2]
    # print(text)

    return text


def train_test_feature(train, test):
    """
    Used to obtained numeric features from the tweets in the form of vector.
    :param train: Training dataframe
    :param test: Testing dataframe
    :return: return vectorized form of training and testing data.
    """
    train_feature = []
    test_feature = []
    words_vector = CountVectorizer()

    for value in train['clean_text']:
        train_feature.append(value)

    for value in test['clean_text']:
        test_feature.append(value)

    train_vector = words_vector.fit_transform(train_feature)

    test_vector = words_vector.transform(test_feature)

    return train_vector, test_vector


def zero_classifier(train, test):
    """
    This is a base line model which predicts everything as majority class. Here majority class is 0.
    :param train:
    :param test:
    :return:
    """

    result = 0

    for value in test['target']:
        if value == 4:
            result += 1

    print(result/len(test['target']))


def bow(train, test):
    """
    Bag of word vectorization with Random Forest classifier.
    :param train: training data
    :param test: testing data
    :return: None
    """

    train_feature, test_feature = train_test_feature(train, test)

    forest = RandomForestClassifier(n_estimators=100)

    model = forest.fit(train_feature.toarray(), train['target'])
    prediction = model.predict(test_feature.toarray())
    print(accuracy_score(prediction, test['target']))


def naive_bayes(train, test):
    """
    Naive Bayes classifier
    :param train: training data
    :param test: testing data
    :return: None
    """
    train_feature, test_feature = train_test_feature(train, test)

    bayes = GaussianNB()

    model = bayes.fit(train_feature.toarray(), train['target'])
    prediction = model.predict(test_feature.toarray())
    print(accuracy_score(prediction, test['target']))


def main():
    """
    This function is used to read the csv into a data frame. Cleaning of the data and finally classify the updated data
    into positive or negative.
    :return: None
    """
    columns = ['target', 'ids', 'date', 'flag', 'user', 'normal_text']
    frame = pd.read_csv('dataset.csv', encoding='latin-1')
    frame.columns = columns

    frame = drop_columns(['ids', 'date', 'flag', 'user'], frame)
    frame.head()

    sns.countplot(x='target', data=frame)
    plt.show()

    clean_text = []

    for index, row in frame.iterrows():

        clean_text.append(cleaning(row['normal_text']))

    word_cloud(clean_text)

    frame['clean_text'] = clean_text

    train, test = train_test_split(frame, test_size=0.2, random_state=30)

    zero_classifier(train, test)
    naive_bayes(train, test)
    bow(train, test)


if __name__ == '__main__':
    main()
