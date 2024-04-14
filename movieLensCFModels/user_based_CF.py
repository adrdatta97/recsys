# imports
import numpy as np
import pandas as pd
from libreco.data import random_split, DatasetPure
from libreco.algorithms import UserCF
from libreco.evaluation import evaluate
from utils.data_loader import kaggle_data_loader
from utils.recommend import movie_recommender

def main():
    # load data
    movie_lens_data = kaggle_data_loader()
    movie_lens_data = movie_lens_data.head(1000000)
    movie_lens_data = movie_lens_data.rename(columns={
        "userId": "user",
        "movieId": "item",
        "rating": "label"
    })

    # split data into train set and test set
    train_data, test_data = random_split(
        movie_lens_data, multi_ratios=[0.8, 0.2] 
        )

    train_data, data_info = DatasetPure.build_trainset(
        train_data
    )

    test_data = DatasetPure.build_testset(
        test_data
    )

    print(data_info)

    # compile user-based model
    user_based_model = UserCF(task="rating",
                            data_info=data_info,
                            sim_type="cosine",
                            k_sim=10,
                            store_top_k=True
                            )

    # train model
    user_based_model.fit(
        train_data,
        verbose=2,
        neg_sampling=False,
        metrics=["loss", "rmse", "mae"]
    )

    # evaluate model on test set
    test_metrics = evaluate(
        model = user_based_model,
        data = test_data,
        neg_sampling=True,
        metrics=["loss", "rmse", "mae"]
    )

    # print model metrics on test set
    print(f"Test metrics: {test_metrics}")

    # grab movie recommendations for user 296 in the format {user_id: np.array(movie_ids)}
    x = user_based_model.recommend_user(
        user=296, n_rec=10
    )

    # map movie ids to movie titles
    movie_recommender(
        x
    )

if __name__=="__main__":
    main()






