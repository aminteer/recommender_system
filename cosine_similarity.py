import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample matrix of user data vs. item ratings (users in rows, items in columns)
ratings_matrix = np.array([
    [3, 3, 4, 0, 4, 2, 3, 0],
    [3, 5, 4, 3, 3, 0, 0, 4],
    [0, 4, 0, 5, 0, 0, 2, 1],
    [2, 0, 0, 4, 0, 4, 4, 5],
])

# Normalize ratings by replacing zeros with the mean rating for each user
mean_user_rating = ratings_matrix.mean(axis=1, keepdims=True)
ratings_diff = np.where(ratings_matrix != 0, ratings_matrix, mean_user_rating)

# Convert ratings to binary (1 if rating >= 1, 0 otherwise)
binary_ratings = (ratings_matrix >= 1).astype(int)

# Calculate cosine similarity
#cosine_sim = cosine_similarity(ratings_diff)
cosine_sim = cosine_similarity(binary_ratings)

print("Cosine similarity matrix:")
print(cosine_sim)

###quiz solution
# def cos(a,b):
#     return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

# simmat=np.zeros((4,4))
# for i in range(4):
#     for j in range(4):
#         simmat[i,j] = cos(Ub[i],Ub[j])
#         if i<j:
#             print(users[i]+'-'+users[j], cos(Ub[i],Ub[j]))
# print(simmat)     