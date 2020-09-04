import pandas as pd
import numpy as np
import os
import re
from flair.models import TextClassifier
from flair.data import Sentence
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# sentiment analysis

def get_offensive_corpus(path):
    data = pd.read_csv(path)
    offensive_words = dict()
    for i in data.columns:
        offensive_words[i] = set(data[i].dropna())
    cols = ['offensive', 'sexual content', 'racism', 'violence']
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            offensive_words[cols[i]] = offensive_words[cols[i]
                                                       ].difference(offensive_words[cols[j]])

    def process(words):
        word_list = []
        for i in words:
            text = i.strip()
            text = text.lower()
            text = re.sub(r'[.\-_?!,;:#]', "", text)
            word_list.append(text)
            text = re.sub(r'(.)\1{2,}', r'\1', text)
            word_list.append(text)
        word_list = list(filter(lambda x: len(x) > 0, word_list))
        return set(word_list)

    offensive_words = {i: process(j) for i, j in offensive_words.items()}
    return offensive_words


def initialise(path):
    offensive_path = os.path.join(path, "word_corpus.csv")
    offensive_words = get_offensive_corpus(offensive_path)

    model = TextClassifier.load('en-sentiment')
    profane_model = joblib.load(os.path.join(path, 'model.joblib'))
    profane_vectorizer = joblib.load(os.path.join(path, 'vectorizer.joblib'))

    return model, offensive_words, profane_model, profane_vectorizer


def preprocess(text):
    text = text.lower()
    text = text.strip()
    text = re.sub(r"([a-z]{2,})[\.?,]([a-z]+)", "\\1 \\2", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text.strip()
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return text


def check_offense(text, offensive_words, model):
    sentence = Sentence(text)
    model.predict(sentence)
    labels = sentence.labels

    text = preprocess(text)
    intersect = {key: {word for word in words if re.search(
        r"\b{}\b".format(word), text)} for key, words in offensive_words.items()}
    if 'NEGATIVE' in str(labels[0]):
        return intersect, 'NEGATIVE'
    else:
        return intersect, 'POSITIVE'


def intersect(offensive_dict):
    max_cat = max(offensive_dict.items(), key=lambda x: len(x[1]))
    offensive = 1 if len(max_cat[1]) > 0 else 0
    offense_cat = max_cat[0] if offensive == 1 else ""
    return offensive, offense_cat


def _get_profane_prob(prob):
    return prob[1]


def calculate(profane_prob, flair_label, vader_label, offensive_words_detected):
    notify = 0
    new_prob = profane_prob
    if offensive_words_detected:
        new_prob = profane_prob+(1-profane_prob)*0.5
    if flair_label == 'POSITIVE' and vader_label == 'POSITIVE':
        new_prob = new_prob/2
        if offensive_words_detected:
            notify = 1
    return notify, new_prob


def sentiment_combine(vader_label, flair_label):
    if vader_label == 'POSITIVE' and flair_label == 'POSITIVE':
        return 'POSITIVE'
    elif vader_label == 'NEGATIVE' and flair_label == 'NEGATIVE':
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'


def predict(file, model, offensive_words, upfolder, profane_model, profane_vectorizer):
    try:
        data = pd.read_csv(file)
    except UnicodeDecodeError:
        data = pd.read_csv(file, encoding='cp1252')
    except Exception:
        return "Ensure your file can be read"

    try:
        data['intersect'], data['flair_label'] = zip(
            *data.apply(lambda x: check_offense(x['text'], offensive_words, model), axis=1))

        data['offensive'], data['sub_category'] = zip(
            *data['intersect'].apply(lambda x: intersect(x)))

        data['profane_check_prob'] = np.apply_along_axis(
            _get_profane_prob, 1, profane_model.predict_proba(profane_vectorizer.transform(data['text'])))

        analyzer = SentimentIntensityAnalyzer()
        data['vader_score'] = data['text'].apply(
            lambda x: analyzer.polarity_scores(x))
        data['vader_label'] = data['vader_score'].apply(
            lambda x: 'POSITIVE' if x['compound'] > 0 else 'NEGATIVE')
        data['notify_moderator'], data['profane_probability'] = zip(*data.apply(lambda x: calculate(
            x['profane_check_prob'], x['flair_label'], x['vader_label'], x['offensive']), axis=1))

        data['sentiment'] = data.apply(lambda x: sentiment_combine(
            x['vader_label'], x['flair_label']), axis=1)

        data[['text', 'sentiment', 'profane_probability', 'sub_category', 'notify_moderator']].to_csv(
            os.path.join(upfolder, "result.csv"), index=False)
        return ""
    except Exception:
        return "Error encountered at execution. Kindly contact the dev team at coeaisd@cet.edu.in"


def predict_text(text, model, offensive_words, profane_model, profane_vectorizer):
    try:
        inter, flair_label = check_offense(text, offensive_words, model)
        offensive, sub_category = intersect(inter)
        profane_check_prob = np.apply_along_axis(
            _get_profane_prob, 1, profane_model.predict_proba(profane_vectorizer.transform([text])))
        analyzer = SentimentIntensityAnalyzer()
        vader_score = analyzer.polarity_scores(text)
        vader_label = 'POSITIVE' if vader_score['compound'] > 0 else 'NEGATIVE'
        _, profane_probability = calculate(
            profane_check_prob, flair_label, vader_label, offensive)

        profane_probability = round(profane_probability[0], 2)
        sentiment = sentiment_combine(vader_label, flair_label)

        if sub_category:
            return (f"Sentiment: {sentiment.lower()}  Profane Probability: {profane_probability}  Sub Category: {sub_category}")
        else:
            return (f"Sentiment: {sentiment.lower()}  Profane Probability: {profane_probability}")
    except Exception:
        return ""


# if __name__ == '__main__':
#     text = take text input
#     model, offensive_words, profane_model, profane_vectorizer = initialise("./")
#     result = predict_text(text,model,offensive_words,profane_model,profane_vectorizer)
