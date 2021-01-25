import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import preprocessor as prepro
import json
from os import path


def clean_review(review):
    contents = review.lower()
    # May want to change these
    prepro.set_options(
        prepro.OPT.URL,
        prepro.OPT.EMOJI,
        prepro.OPT.SMILEY,
        prepro.OPT.NUMBER
    )
    clean_contents = prepro.clean(contents)
    contents = clean_contents
    return contents


def tokenize(review):
    tokens = word_tokenize(review)
    stop_words = set(stopwords.words("english"))
    useful_tokens = []
    lemma = WordNetLemmatizer()
    for token in tokens:
        if (not token in stop_words) and (token.isalpha()):
            lemmatnised_token = lemma.lemmatize(token)
            useful_tokens.append(lemmatnised_token)
    return useful_tokens


def main(reviews):
    # nltk.download()
    processed_reviews = []
    for review in reviews:
        review = clean_review(review)
        tokens = tokenize(review)
        processed_reviews.append({"tokens": tokens})
    with open("../processed_reviews.txt", "w") as f:
        f.write(json.dumps(processed_reviews))
