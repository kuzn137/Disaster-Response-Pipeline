# Disaster Response Pipeline Project

## Author: Inga Kuznetsova

## Project Outline


In this project I have built up a Web application which takes input messages and direct to a particular organization that can give help. The application utilizes a ML model to classify any new messages, and the code trains model. 

*** Files Description

Data .csv files, .db file and model .pkl files are excluded because of large size.

Data: process_data.py: This code takes as its input csv files containing message data and message categories (labels), and creates SQLite database containing a merged and cleaned version of this data

Models: This folder contains the ML pipeline (the pickle object was excluded because of size), train_classifier.py: This code takes the SQLite database produced by process_data.py as an input and uses the data contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.

App: This folder contains run.py and templates which are used to run the main web application.

Also 2 notebooks: ‘ETL Pipeline Preparation.ipynb’, which clean and prepare data and ‘ML Pipeline Preparation.ipynb’ which train and use models are included.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Site Screenshot

![alt text](https://github.com/kuzn137/Disaster-Response-Pipeline/blob/master/Screenshot_webapp.png?raw=true)

### License

Copyright 2020 Inga Kuznetsova
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

