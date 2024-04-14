import pandas as pd
import numpy as np
from utils.data_loader import kaggle_data_loader, kaggle_data_loader_recommend

def user_has_rated(userId, merged_df):
    return merged_df.loc[merged_df['userId']==userId, ['title', 'rating', 'genres']].iloc[0]

def movie_recommender(recommendation_dict):
    rating_df = kaggle_data_loader()
    movie_df = kaggle_data_loader_recommend()
    merged_df = pd.merge(rating_df, movie_df, on='movieId')
    print(merged_df.columns)
    
    userIds = list(recommendation_dict.keys())
    for user in userIds:
        user_has_previously_rated = user_has_rated(user, merged_df)
        print(f"user {user} has previously rated: {user_has_previously_rated}")

    for user in recommendation_dict:   
        print(f'Movie recommendations for user: {user}')
        for movie_id in recommendation_dict[user]:
            print(merged_df.loc[merged_df['movieId'] == movie_id, ['title', 'genres']].iloc[0])


# check function
# test_user = {
#     296: np.array([480, 592, 356, 377, 349, 165, 364, 500, 316, 589])
# }

# movie_recommender(test_user)