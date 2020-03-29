import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
import joblib
import sys
import argparse
# import nltk
import os

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

np.random.seed(500)


def preprocess_text(text):
    # Step - a : Remove blank rows if any.
    text = os.linesep.join([s for s in text.splitlines() if s])
    # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    text = text.lower()
    # Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
    text = word_tokenize(text)
    # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.# WordNetLemmatizer requires
    # Postags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    # Declaring Empty List to store the words that follow the rules for this step
    final_words = []
    # Initializing WordNetLemmatizer()
    word_lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(text):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_final = word_lemmatized.lemmatize(word, tag_map[tag[0]])
            final_words.append(word_final)

    # The final processed set of words for each iteration will be stored in 'text_final'
    text = str(final_words)
    return text


def main(argv):
    parser = argparse.ArgumentParser(description='Predict text category')
    parser.add_argument('-m', action='store', default='finalized_model_svm.sav', dest='model',
                        help='filepath to finalized model', nargs='?')
    parser.add_argument('-v', action='store', default='tfidf_vectorizer.sav', dest='vectorizer',
                        help='filepath to tfidf vectorizer', nargs='?')
    parser.add_argument('-e', action='store', default='label_encoder.sav', dest='encoder',
                        help='filepath to label encoder', nargs='?')
    parser.add_argument('-i', action='store', dest='input', help='input text to categorize', default=None, nargs='*')

    args = parser.parse_args(argv)
    finalised_model_filename = args.model
    tfidf_vectorizer_filename = args.vectorizer
    label_encoder_filename = args.encoder
    input_text = args.input

    for comment in input_text:
        preprocessed_text = preprocess_text(comment)
        tfidf_vect = joblib.load(tfidf_vectorizer_filename)
        predict_x_tfidf = tfidf_vect.transform([preprocessed_text])

        # load the model from disk
        loaded_model = joblib.load(finalised_model_filename)
        result = loaded_model.predict(predict_x_tfidf)
        # print(result)

        label_encoder = joblib.load(label_encoder_filename)
        print(label_encoder.inverse_transform(result)[0])


if __name__ == "__main__":
    main(sys.argv[1:])
