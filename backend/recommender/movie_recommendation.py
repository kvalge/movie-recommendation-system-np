from backend.data.load_data import load_data
import numpy as np


def movie_recommendation_algorithm(user_index, top_n=5, num_neighbors=5, max_diff=2):
    """
    Recommends movies to a selected user based on similar users' viewing history and ratings.

    The algorithm works as follows:
    1. Finds users (neighbors) who have rated the same movies as the target user,
       where the difference in ratings is within a specified threshold (`max_diff`).
    2. Selects the top `num_neighbors` most similar users based on the number of such shared ratings.
    3. Recommends movies that the similar users have rated, which the target user has not seen,
       excluding movies rated poorly (rating ≤ 2) by those users.
    4. Scores recommended movies based on the sum of the neighbors' ratings and returns the top `top_n` movies.

    Parameters:
        user_index (int): The index of the user for whom to generate recommendations.
        top_n (int): The number of movies to recommend.
        num_neighbors (int): How many similar users to consider.
        max_diff (int): The maximum allowed difference in ratings for a movie to be considered commonly rated.

    Returns:
        np.ndarray: Array of recommended movie titles.
    """
    users, movies, ratings = load_data()

    target_ratings = ratings[user_index]
    seen = target_ratings > 0

    similarities = []
    for other_index in range(ratings.shape[0]):
        if other_index == user_index:
            continue
        other_ratings = ratings[other_index]

        both_rated = np.logical_and(seen, other_ratings > 0)
        rating_diff = np.abs(target_ratings - other_ratings)
        valid_common = np.logical_and(both_rated, rating_diff <= max_diff)

        similarity = np.sum(valid_common)
        if similarity > 0:
            similarities.append((other_index, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_users = [idx for idx, _ in similarities][:num_neighbors]

    scores = np.zeros(ratings.shape[1])
    for neighbor in top_users:
        neighbor_ratings = ratings[neighbor]
        for i in range(ratings.shape[1]):
            if not seen[i] and neighbor_ratings[i] > 2:
                scores[i] += neighbor_ratings[i]

    recommended_indices = scores.argsort()[::-1][:top_n]
    return movies[recommended_indices]


if __name__ == '__main__':
    algorithm = movie_recommendation_algorithm(0)
    for title in algorithm:
        print("-", title)
