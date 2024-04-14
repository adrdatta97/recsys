import pandas as pd
# Kaggle dataset URL: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset
def data_loader_rating():
    ratings_df = pd.read_csv('kaggle_data/Ratings.csv')
    print(ratings_df.head())
    return ratings_df

def data_loader_recommender():
    books_df = pd.read_csv('kaggle_data/Books.csv')
    return books_df
