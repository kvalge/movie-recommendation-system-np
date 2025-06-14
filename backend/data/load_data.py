from pathlib import Path

import numpy as np

def load_data(filename="movie_viewing_data.npz"):
    """
    Loads user, movie, and rating data from a compressed NPZ file.

    Parameters:
        filename (str): Path to the .npz file containing the data.

    Returns:
        tuple: (users, movies, ratings) where:
            - users (ndarray): Array of user IDs.
            - movies (ndarray): Array of movie titles.
            - ratings (ndarray): 2D array of ratings (shape: [num_users, num_movies]).
    """
    base_dir = Path(__file__).parent
    full_path = base_dir / filename

    data = np.load(full_path, allow_pickle=True)
    users = data['users']
    movies = data['movies']
    ratings = data['ratings']

    return users, movies, ratings

if __name__ == '__main__':
    users, movies, ratings = load_data()
    print("Users shape:", users.shape)
    print("Movies shape:", movies.shape)
    print("Ratings shape:", ratings.shape)
    print("Sample ratings:\n", ratings[:5])
