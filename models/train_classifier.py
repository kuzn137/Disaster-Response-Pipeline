import sys
# import libraries
import nltk
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from sklearn.model_selection import GridSearchCV
import nltk
nltk.download('punkt')  
nltk.download('stopwords')
# import libraries
import nltk
import pickle
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from sklearn.model_selection import GridSearchCV
import nltk
nltk.download('punkt')  
nltk.download('stopwords')
import pandas as pd
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine
from sklearn.preprocessing import OneHotEncoder
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
import re
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
import warnings
warnings.simplefilter('ignore')
from sklearn.metrics import  f1_score,  accuracy_score, classification_report, fbeta_score, make_scorer

#database_filepath = 'sqlite:///./data/DisasterResponse.db'
def load_data(database_filepath):
    """Function to load data for the model.
       Args: 
          database_path
       Returns: 
          pd.dataframes: incoming features vector X, outcome vector Y; list: categories names
    """
    engine = create_engine('sqlite:///./data/DisasterResponse.db')
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)
    #exclude colums that are not needed in model
    col=[i for i in df.columns if i not in ['id','original', 'genre']]
    X = df["message"]
    Y = df.iloc[:,4:]
    #global category_names
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """Function tokenize text.
       Args: 
          messages text
       Returns: 
         Tokens
    """
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    
def evaluate_model(model, X_test, y_test, category_names):
    """Function to print model evalution scores, comparing real test data and predicted
         Args: 
           model, incoming features dataframe X_test, test labeles dataframe y_test, list of category names
         Returns:     
           None
    """
    y_pred = model.predict(X_test)
    labels = np.unique(y_pred)
    print(labels)
    #print out score for each class and mean scores, including precision, recall, f1 score
    print(classification_report(y_test.values, y_pred, target_names=category_names.values))

def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize))
    , ('tfidf', TfidfTransformer())
    , ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {'vect__min_df': [1, 5],
             # 'tfidf__use_idf':[True, False],
              'clf__estimator__n_estimators':[50, 100], 
              'clf__estimator__min_samples_split':[5],
              #'vect__max_features': (5000, 10000)
       
                 }

    #cv = GridSearchCV(estimator=pipeline, param_grid=parameters, verbose=3)   
    #my_scorer = make_scorer(f1_score(y_test, y_pred, average='macro'), greater_is_better=True)
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring="f1_weighted")


    return cv
   

    

def save_model(model, model_filepath):
    """Function saves model to pickle file.
         Args: 
           model, path where to save model
         Returns:     
           None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()