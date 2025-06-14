import numpy as np


def generate_dummy_data(num_users=100, num_movies=35, sparsity=0.7, seed=42):
    """
    Generates a synthetic movie rating dataset for testing recommendation systems.

    Parameters:
        num_users (int): Number of users to simulate.
        num_movies (int): Number of movies to include.
        sparsity (float): Fraction of missing ratings (0 = fully rated, 1 = fully sparse).
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (users, movies, ratings) where:
            - users (ndarray): Array of user IDs.
            - movies (ndarray): Array of movie titles (length = num_movies).
            - ratings (ndarray): 2D array of shape (num_users, num_movies)
                                 with integer ratings from 1 to 5, or 0 for missing ratings.
    """
    np.random.seed(seed)

    movies = np.array([
        "Shutter Island",
        "The Godfather",
        "Last Call",
        "The Usual Suspects",
        "Pulp Fiction",
        "City of Ghosts",
        "Forrest Gump",
        "Inception",
        "Fight Club",
        "The Matrix",
        "Heat",
        "Seven",
        "Psycho",
        "The Silence of the Lambs",
        "Pan's Labyrinth",
        "Nipernaadi",
        "A Beautiful Mind",
        "Dogville",
        "Memento",
        "Joker",
        "Balibo",
        "Stand Up Guys",
        "Life Is Beautiful",
        "Thelma & Louise",
        "Parasite",
        "Volver",
        "Hannibal",
        "Man on Fire",
        "Spy Game",
        "Lost in Translation",
        "Another Round",
        "Kinds of Kindness",
        "The Favorite",
        "Poor Things",
        "The Lobster"
    ], dtype="<U100")

    num_movies = min(num_movies, len(movies))
    movies = movies[:num_movies]

    users = np.arange(1, num_users + 1)
    ratings = np.zeros((num_users, num_movies), dtype=int)

    prob_matrix = np.random.rand(num_users, num_movies)
    random_scores = np.random.randint(1, 6, size=(num_users, num_movies))

    mask = prob_matrix < (1 - sparsity)
    ratings[mask] = random_scores[mask]

    return users, movies, ratings

if __name__ == '__main__':
    generate_dummy_data()

