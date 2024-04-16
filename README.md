# Collaborative Filtering based Recommender System
_Completed in partial fulfillment of Master of Science - Data Analytics degree from Queen Mary University of London._

This code is supplementary to the master’s thesis on recommender systems. Link to the full thesis: [Link](https://drive.google.com/file/d/1tnwfL5aqheLoPntLSuVgCsPFs_2DXus1/view?usp=share_link) 

In this work, two recommender system algorithms are implemented and analyzed on the MovieLens and Book datasets. The data was collected from the popular data science competition platform Kaggle. The dataset links are provided below :

* [MovieLens Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?datasetId=339&sortBy=voteCount&select=tag.csv)

* [Book Recommendation system](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset?select=Users.csv)

Two algorithms are explored in this work - the User Based Collaborative Filtering model and the Item Based Collaborative Filtering model. This is a comparative study between the recommendations provided by each algorithm on the two datasets, the results of which can be found below. Root Mean Squared Error and Mean Absolute Error are the performance metrics used in this work to evaluate performance of the models.

The algorithm implementation is provided by the open-source LibRecommendation library [Link](https://librecommender.readthedocs.io/en/latest/#:~:text=LibRecommender%20is%20an%20easy%2Dto,different%20kinds%20of%20recommendation%20models.) , which provides implementations of most common recommender systems available in the literature. Due to a lack of cloud based hardware accelerators (GPUs, TPUs) a subset of the data was used. The results indicate that performance of the models were not deeply affected by this choice.

The code is written is Python, and the directory structure of this repository is as follows:
* Exploratory Data Analysis notebooks :
  - Books_EDA.ipynb
  - Movies_EDA_FINAL.ipynb
* BookRecommendationCFModels :
  - item_based_CF.py
  - user_based_CF.py
  - utils:
    - data_loader.py
    - recommender.py
  - kaggle_data :
    - Ratings.csv
    - Books.csv
* movieLensCFModels :
  - item_ based_CF.py
  - user_based_CF.py
  - data_kaggle:
    - movie.csv
    - ratings.csv
  - utils:
    - data_loader.py
    - recommend.py
* requirements.txt

# Results : 

DATASET | USED BASED MODEL | ITEM BASED MODEL
| :--- | ---: | :---:
BOOKS  | RMSE: 4.351 <br> MAE: 3.586 | RMSE: 3.824 <br> MAE: 3.047
MOVIELENS  | RMSE: 2.967 <br> MAE: 2.859 | RMSE: 0.872 <br> MAE: 0.657




