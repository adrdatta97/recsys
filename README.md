Recommendation Systems (Collaborative Filtering)

In this work , I've trained and tested two recommendation system algorithms : user- based and item-based CF models on two different datasets(MovieLens, Book) to explore the performance in recommending users relevant content.
The entire dataset was first downloaded from Kaggle and loaded into a pandas dataframe, where only the subset of a dataset was selected for training and testing of the models.
The models are written using the LibRecommen- dation library. This is an open source recommender system library that provide TensorFlow, PyTorch, Cython implementations of major recommender algorithms in the industry.

In Movies_EDA_Final.ipynb and Books_EDA.ipynb one can find the exploratory data analysis. 
In both the files we can see WordClouds, histograms, top rated movies, most famous genres - each of them to help us visualize different aspects of the niche. 
In movieLensCFModels > item_based_CF.py  -> Implementation of the Item-based model
In movieLensCFModels > user_based_CF.py -> Implementation of the User-based model
In bookRecommendations > item_based_CF.py -> Implementation of the Item-based model
In bookRecommendations > user_based_CF.py -> Implementation of the User-based model

In the Movie Lens dataset, the user-based recommendation system performs reasonably well, with an Root Mean Squared Error(RMSE) of 2.967 and Mean Absolute Error(MAE) of 2.859. 
However, the item-based approach significantly outperforms it, achieving a much lower RMSE (0.872) and MAE (0.657). 
In the Books Recommendation System, the item-based recommendation system outperforms the user-based approach in terms of prediction accuracy. 
The item-based model achieves a lower RMSE (3.824) and MAE (3.047) compared to the user-based model with an RMSE of 4.351 and MAE of 3.586.

Across the datasets, we can see that the item- based model outperformed the user based model, supporting the reports from the existing literature.
It was also seen that excluding implicit feedback data points improved performance in both the models in the book recommenda- tions dataset.
