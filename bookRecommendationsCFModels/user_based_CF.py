# imports
from libreco.algorithms import UserCF
from libreco.data import DatasetPure, random_split
from libreco.evaluation import evaluate

from utils.data_loader import data_loader_rating
from utils.recommender import book_recommender

def predict(user_based_model, user_id):
    # model book recommendations for user 276725 in the format { user_id: np.array(ISBN)}
    x = user_based_model.recommend_user(
        user=user_id, n_rec=5 
    )
    print(x)

    # map book ISBNs to book titles
    print(book_recommender(
        x
    ))


def main():
    # load data
    books_data = data_loader_rating()
    books_data = books_data.rename(columns={
        "User-ID": "user",
        "ISBN": "item",
        "Book-Rating": "label"
    })

    # droppings rows which have rating 0, which indicates implicit feedback (source: Kaggle)
    books_data = books_data.drop(books_data.query('label == 0').index)

    # split dataset into training and testing data
    train_data, test_data = random_split(
        books_data, multi_ratios=[0.8, 0.2] 
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
    user_based_model = UserCF(
                            task="rating",
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

    # evaluate model performance on test set
    test_metrics =  evaluate(
        model = user_based_model,
        data = test_data,
        neg_sampling=False,
        metrics=["loss", "rmse", "mae"]
    )

    # print metrics
    print(f"Test metrics: {test_metrics}")
    return user_based_model
    # # model book recommendations for user 276725 in the format { user_id: np.array(ISBN)}
    # x = user_based_model.recommend_user(
    #     user=122881, n_rec=5 
    # )
    # print(x)

    # # map book ISBNs to book titles
    # book_recommender(
    #     x
    # )

if __name__=="__main__":
    model = main()
    predict(model, user_id=276729)