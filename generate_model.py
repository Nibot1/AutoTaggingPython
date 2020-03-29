import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import joblib
import nltk
import timeit

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

np.random.seed(500)


def preprocess_data(csv_filename):
    corpus = pd.read_csv(r"" + csv_filename, encoding='latin-1')
    # Step - a : Remove blank rows if any.
    corpus['text'].dropna(inplace=True)
    # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    corpus['text'] = [entry.lower() for entry in corpus['text']]
    # Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
    corpus['text'] = [word_tokenize(entry) for entry in corpus['text']]
    # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting. WordNetLemmatizer requires
    # Postags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    for index, entry in enumerate(corpus['text']):
        # Declaring Empty List to store the words that follow the rules for this step
        final_words = []
        # Initializing WordNetLemmatizer()
        word_lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_final = word_lemmatized.lemmatize(word, tag_map[tag[0]])
                final_words.append(word_final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        corpus.loc[index, 'text_final'] = str(final_words)
    return corpus


def main():
    corpus = preprocess_data("corpus.csv")

    # Generate test and train data from preprocessed Data
    train_x, test_x, train_y, test_y = model_selection.train_test_split(corpus['text_final'], corpus['label'],
                                                                        test_size=0.3)
    # Transform labels
    encoder = LabelEncoder()
    encoder.fit(train_y)
    train_y = encoder.transform(train_y)
    test_y = encoder.transform(test_y)
    joblib.dump(encoder, "label_encoder.sav")
    print("saved label_encoder")

    # Generate Tfidf-vector
    tfidf_vect = TfidfVectorizer(max_features=5000)
    tfidf_vect.fit(corpus['text_final'])
    joblib.dump(tfidf_vect, "tfidf_vectorizer.sav")
    print("saved tfidf_vectorizer")
    # Transform Tfidf-vector to Tfidf
    train_x_tfidf = tfidf_vect.transform(train_x)
    test_x_tfidf = tfidf_vect.transform(test_x)

    print(tfidf_vect.vocabulary_)

    print(train_x_tfidf)

    # fit the training dataset on the NB classifier
    naive = naive_bayes.MultinomialNB()
    naive.fit(train_x_tfidf, train_y)  # predict the labels on validation dataset
    predictions_nb = naive.predict(test_x_tfidf)  # Use accuracy_score function to get the accuracy
    print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_nb, test_y) * 100)

    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    svm_model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    svm_model.fit(train_x_tfidf, train_y)  # predict the labels on validation dataset
    predictions_svm = svm_model.predict(test_x_tfidf)  # Use accuracy_score function to get the accuracy
    print("SVM Accuracy Score -> ", accuracy_score(predictions_svm, test_y) * 100)

    # save the model to disk
    filename = 'finalized_model_naive.sav'
    joblib.dump(naive, filename)
    print("saved naive model")

    filename = 'finalized_model_svm.sav'
    joblib.dump(svm_model, filename)
    print("saved svm model")


print(timeit.timeit(main, number=1))
