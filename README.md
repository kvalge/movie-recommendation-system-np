# Movie Recommendation System with NumPy and Flask  

The recommendation algorithm suggests movies to a user based on ratings from other users with similar tastes.  
It works by:  
Finding users who have rated many of the same movies as the target user, and whose ratings are close (difference ≤ 2) on those movies.  
Selecting the top similar users (neighbors) to consider.
Recommending movies that these neighbors liked (ratings above 2), which the target user has not yet rated.
Ranking these recommended movies by the sum of neighbors’ ratings to provide the top suggestions.  

## Data
The synthetic movie rating data is generated using `data/generate_data.py`.  
The generated data is saved in a compressed NumPy `.npz` file via `data/save_data.py`.  
Data is load with `data/load_data.py`.  
Ratings matrix of data consists users as rows and columns as movies, every element of matrix is rating from 1 to 5 and 0 when user hasn't rated the movie.  

## Technologies
Python 3.11  
Numpy  
Flask  

### Acknowledgement
ChatGPT GPT-4o is used for code generation.  