import pandas as pd
from utils.data_loader import data_loader_rating, data_loader_recommender

def user_has_rated(userId, merged_df):
    return merged_df.loc[merged_df['User-ID']==userId, ['Book-Title', 'Book-Rating']].iloc[0]

def book_recommender(recommendation_dict):
    rating_df = data_loader_rating()
    books_df = data_loader_recommender()
    merged_df = pd.merge(rating_df, books_df, on='ISBN')
    print(merged_df.columns)
    
    userIds = list(recommendation_dict.keys())
    for user in userIds:
        user_has_previously_rated = user_has_rated(user, merged_df)
        print(f"user {user} has previously rated: {user_has_previously_rated}")

    for user in recommendation_dict:
        print(f'Book recommendations for user: {user}')
        for book_id in recommendation_dict[user]:
            print(merged_df.loc[merged_df['ISBN'] == book_id, 'Book-Title'].iloc[0])


# check function
# book_recommender(
#     {
#     276727: ['0913780146', '0843108118', '0807072125', '0743411366',
#        '0737000465', '0684836556', '0670857661', '0449209881',
#        '0375508627', '0312291523']
#     }
# )