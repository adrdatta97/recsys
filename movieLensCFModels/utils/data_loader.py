import pandas as pd

# Kaggle movie lens dataset URL: https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset

def kaggle_data_loader():
    rating_df = pd.read_csv('data_kaggle/rating.csv')
    return rating_df

def kaggle_data_loader_recommend():
    movie_df = pd.read_csv('data_kaggle/movie.csv')
    return movie_df