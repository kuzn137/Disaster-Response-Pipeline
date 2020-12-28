import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Function to load data.
       Args: 
         path to data .csv file with messages, path to data .csv file with categories
       Returns: 
         pd.dataframe with messages and categories
    """
    https://www.linkedin.com/in/inga-kuznetsova-a0b8b521/
    # load messages dataset
    messages = pd.read_csv('./data/disaster_messages.csv')
    # load categories dataset
    categories = pd.read_csv('./data/disaster_categories.csv')
    # ### 2. Merge datasets.
    # - Merge the messages and categories datasets using the common id
    # - Assign this combined dataset to `df`, which will be cleaned in the following steps
    # merge datasets
    df = pd.merge(messages, categories, on = 'id')
    return df

def clean_data(df):
    """Function to clean data for the model.
       Args: 
          pd data frame
       Returns: 
          cleaned pd.dataframe
    """
    # ### 3. Split `categories` into separate category columns.
    # - Split the values in the `categories` column on the `;` character so that each value becomes a separate column. You'll find [this method]      (https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.Series.str.split.html) very helpful! Make sure to set `expand=True`.
    # - Use the first row of categories dataframe to create column names for the categories data.
    # - Rename columns of `categories` with new column names.
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    categories.columns = categories.iloc[1]
    categories.columns =[i[:-2] for i in categories.columns]
    # select the first row of the categories dataframe
    row  = categories.iloc[1]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [i[:-2] for i in row]    
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1] 
        # convert column from string to numeric
        categories[column] =  categories[column].astype('int')
     # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
     # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
     # ### 6. Remove duplicates.
     # drop duplicates
    df=df.drop_duplicates()
    df.related.replace(2,1,inplace=True)
    return df
    

def save_data(df):
    """Function to save data to database.
       Args: 
         pd dataframe
       Returns: 
         None
    """
    engine = create_engine('sqlite:///./data/DisasterResponse.db')
    #saves data to sql database
    df.to_sql('DisasterResponse', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()