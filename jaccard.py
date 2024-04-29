import numpy as np

# Sample matrix of user data vs. item ratings (users in rows, items in columns)
ratings_matrix = np.array([
    [3, 3, 4, 0, 4, 2, 3, 0],
    [3, 5, 4, 3, 3, 0, 0, 4],
    [0, 4, 0, 5, 0, 0, 2, 1],
    [2, 0, 0, 4, 0, 4, 4, 5],
])

# Convert ratings to binary (1 if rating >= 3, 0 otherwise)
binary_ratings = (ratings_matrix >= 3).astype(int)

def jaccard_distance(a, b):
    """Calculate the Jaccard distance between two binary vectors."""
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1  # Completely dissimilar if no items are rated by both users
    return 1 - intersection / union

# Calculate Jaccard distance for each pair of users
n_users = binary_ratings.shape[0]
jaccard_distances = np.zeros((n_users, n_users))

for i in range(n_users):
    for j in range(i, n_users):
        distance = jaccard_distance(binary_ratings[i], binary_ratings[j])
        jaccard_distances[i, j] = distance
        jaccard_distances[j, i] = distance  # the distance matrix is symmetric

print("Jaccard distance matrix:")
print(jaccard_distances)

ratings = np.array([[3, 3, 4, 0, 4, 2, 3, 0],
                    [3, 5, 4, 3, 3, 0, 0, 4],
                    [0, 4, 0, 5, 0, 0, 2, 1],
                    [2, 0, 0, 4, 0, 4, 4, 5]])
Ub = (ratings>0).astype(int)
print(Ub)

def jaccard(a,b):
    return (a*b).sum()/((a+b)>0).sum()

users = ["firechicken","mike0702","zephyros","dadvador"]

simmat=np.zeros((4,4))
for i in range(4):
    for j in range(4):
        simmat[i,j] = jaccard(Ub[i],Ub[j])
        if i<j:
            print(users[i]+'-'+users[j], jaccard(Ub[i],Ub[j]))

print("Jaccard similarity matrix per code Q1:")
print(simmat)

print("Jaccard with good/bad binary with good >=3")
Ur = (ratings>=3).astype(int)
print(Ur)

simmat=np.zeros((4,4))
for i in range(4):
    for j in range(4):
        simmat[i,j] = jaccard(Ur[i],Ur[j])
        if i<j:
            print(users[i]+'-'+users[j], jaccard(Ur[i],Ur[j]))
