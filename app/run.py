import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """Function tokenize text.
		Args: 
			messages text
		
		Returns: 
			Tokens
		"""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """Function creates template with graphs.
        Args: 
         None
        Returns: 
            template with graphs
    """
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # The top 20 categories (from 36) with the highest % of messages 
    top_categ_df = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()/len(df)#.sort_values(ascending = False)[:,0:10]
    top_categ_df=(top_categ_df.sort_values(ascending=False)[0:20])
    top_categ_names = list(top_categ_df.index)
    top_categ_proportions = list(top_categ_df)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    #First graph shows percentage of messages in 20 top categories
    #Second graph shows genre distribution
    graphs = [
        
        {
            'data': [
                Bar(
                    x = top_categ_names,
                    y = top_categ_proportions
                )
            ],
            
            'layout': {
                'title': 'Top 20 Categories by Proportion of Messages Received',
                'yaxis': {
                    'title': "Proportions"
                },
                'xaxis': {
                    'title': "Message Types"
                }}},
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """Function creates template that handels user queries.
         Args: 
           None
         Returns: 
           template that handels user queries.
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()