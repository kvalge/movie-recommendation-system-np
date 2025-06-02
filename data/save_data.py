import os

import numpy as np
from generate_data import generate_dummy_data


def save_data(users, movies, ratings, filename="movie_viewing_data.npz"):
    """
    Saves the user, movie, and rating data to a .npz file.

    Parameters:
        users (ndarray): Array of user IDs.
        movies (ndarray): Array of movie titles.
        ratings (ndarray): 2D array of ratings.
        filename (str): Path where the file will be saved.
    """
    np.savez_compressed(filename, users=users, movies=movies, ratings=ratings)


if __name__ == '__main__':
    users, movies, ratings = generate_dummy_data()
    save_data(users, movies, ratings)
