# imports
from libreco.data import random_split, DatasetPure
from libreco.algorithms import ItemCF
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

    # split dataset into training and testing data
    train_data, test_data = random_split(
        movie_lens_data, multi_ratios=[0.8, 0.2] 
        )

    # create train data object and extract data info
    train_data, data_info = DatasetPure.build_trainset(
        train_data
    )

    # test data 
    test_data = DatasetPure.build_testset(
        test_data
    )

    print(data_info)

    # compile item-based model
    item_based_model = ItemCF(
                            task="rating",
                            data_info=data_info,
                            sim_type="cosine",
                            k_sim=10,
                            store_top_k=True
                            )

    # train model
    item_based_model.fit(
        train_data,
        verbose=2,
        neg_sampling=False,
        metrics=["loss", "precision", "recall"]
    )

    # evaluate model performance on test set
    test_metrics =  evaluate(
        model = item_based_model,
        data = test_data,
        neg_sampling=False,
        metrics=["loss", "rmse", "mae"]
    )

    # print metrics
    print(f"Test metrics: {test_metrics}")

    # model movie recommendations for user 296 in the format { user_id: np.array(movie_id)}
    x = item_based_model.recommend_user(
        user=296, n_rec=10
    )
    print(x)
    # map movie_ids to movie titles
    movie_recommender(
        x
    )

if __name__=="__main__":
    main()