#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.distance import jaccard, cosine 
from pytest import approx
import unittest

from collections import namedtuple
from sklearn.metrics import pairwise_distances

class RecSys():
    def __init__(self,data):
        self.data=data
        self.allusers = list(self.data.users['uID'])
        self.allmovies = list(self.data.movies['mID'])
        self.genres = list(self.data.movies.columns.drop(['mID', 'title', 'year']))
        self.mid2idx = dict(zip(self.data.movies.mID,list(range(len(self.data.movies)))))
        self.uid2idx = dict(zip(self.data.users.uID,list(range(len(self.data.users)))))
        self.Mr=self.rating_matrix()
        self.Mm=None 
        self.sim=np.zeros((len(self.allmovies),len(self.allmovies)))
        
    def rating_matrix(self):
        """
        Convert the rating matrix to numpy array of shape (#allusers,#allmovies)
        """
        ind_movie = [self.mid2idx[x] for x in self.data.train.mID] 
        ind_user = [self.uid2idx[x] for x in self.data.train.uID]
        rating_train = list(self.data.train.rating)
        
        return np.array(coo_matrix((rating_train, (ind_user, ind_movie)), shape=(len(self.allusers), len(self.allmovies))).toarray())


    def predict_everything_to_3(self):
        """
        Predict everything to 3 for the test data
        """
        # Generate an array with 3s against all entries in test dataset
        # your code here
        matrix_shape = len(self.data.test)
        return np.ones(matrix_shape)*3
        
    def predict_to_user_average(self):
        """
        Predict to average rating for the user.
        Returns numpy array of shape (#users,)
        """
        # Generate an array as follows:
        # 1. Calculate all avg user rating as sum of ratings of user across all movies/number of movies whose rating > 0
        # 2. Return the average rating of users in test data
        # your code here
        # get average rating for each user
        # avg_user_rating_idx = self.Mr.sum(axis=1) / (self.Mr != 0).sum(axis=1)
        avg_user_rating_idx = self.data.test.uID.apply(lambda x: self.Mr[self.uid2idx[x]].sum() / (self.Mr[self.uid2idx[x]] != 0).sum())
        #avg_user_rating_idx = avg_user_rating_idx[self.data.test.uID]
        return avg_user_rating_idx
    
    def predict_from_sim(self,uid,mid):
        """
        Predict a user rating on a movie given userID and movieID
        """
        # Predict user rating as follows:
        # 1. Get entry of user id in rating matrix
        # 2. Get entry of movie id in sim matrix
        # 3. Employ 1 and 2 to predict user rating of the movie
        # your code here
        usr_idx = self.uid2idx[uid]
        user_ratings = self.Mr[usr_idx]
        movie_idx = self.mid2idx[mid]
        sim_movie = self.sim[movie_idx]
        #need to divide by count of valid ratings to minimize bias
        pred = np.dot(user_ratings, sim_movie) / np.dot(user_ratings != 0, sim_movie)
        return pred
    
    def predict(self):
        """
        Predict ratings in the test data. Returns predicted rating in a numpy array of size (# of rows in testdata,)
        """
        # your code here
        test_preds = []
        for i in range(len(self.data.test)):
            test_preds.append(self.predict_from_sim(self.data.test.uID[i], self.data.test.mID[i]))
        return np.array(test_preds)
    
    def rmse(self,yp):
        yp[np.isnan(yp)]=3 #In case there is nan values in prediction, it will impute to 3.
        yt=np.array(self.data.test.rating)
        return np.sqrt(((yt-yp)**2).mean())

    
class ContentBased(RecSys):
    def __init__(self,data):
        super().__init__(data)
        self.data=data
        self.Mm = self.calc_movie_feature_matrix()  
        
    def calc_movie_feature_matrix(self):
        """
        Create movie feature matrix in a numpy array of shape (#allmovies, #genres) 
        """
        # your code here
        # get movie features, genres is created from all movies but drops the mID, title, and year columns leaving only the features
        movie_features = np.array(self.data.movies[self.genres])
        return movie_features
    
    def calc_item_item_similarity(self):
        """
        Create item-item similarity using Jaccard similarity
        """
        # Update the sim matrix by calculating item-item similarity using Jaccard similarity
        # Jaccard Similarity: J(A, B) = |A∩B| / |A∪B| 
        # your code here
        # get feature matrix by genre
        feature_matrix = self.Mm
        # calculate jaccard similarity for all pairs of items
        # These features are already binary
                
        def jaccard_similarity(a, b):
            """Calculate the Jaccard distance between two binary vectors."""
            intersection = np.logical_and(a, b).sum()
            union = np.logical_or(a, b).sum()
            if union == 0:
                return 0  # Completely dissimilar if no items are rated by both users
            return float(intersection) / union
        
        # def jaccard_similarity(list1, list2):
        #     intersection = len(list(set(list1).intersection(list2)))
        #     union = (len(list1) + len(list2)) - intersection
        #     return 1 - float(intersection) / union            

        # Calculate Jaccard distance for each pair of users
        n_movies = feature_matrix.shape[0]
        jaccard_similarities = np.zeros((n_movies, n_movies))

        # for i in range(n_movies):
        #     for j in range(i, n_movies):
        #         similarity = jaccard_similarity(feature_matrix[i], feature_matrix[j])
        #         jaccard_similarities[i, j] = similarity
        #         jaccard_similarities[j, i] = similarity  # the distance matrix is symmetric        
        
        #Mm_trace = np.trace(jaccard_similarities) #for debugging
        #Mm_trace = np.trace(final_jaccard) #for debugging
        # Using pairwise_distances is much faster, we will keep this active
        jaccard_similarities = 1 - pairwise_distances(feature_matrix, metric = 'jaccard')
        #compare_trace = np.trace(jaccard_similarities) #for debugging

        self.sim = jaccard_similarities
        
        
                
class Collaborative(RecSys):    
    def __init__(self,data):
        super().__init__(data)
        
    def calc_item_item_similarity(self, simfunction, *X):  
        """
        Create item-item similarity using similarity function. 
        X is an optional transformed matrix of Mr
        """    
        # General function that calculates item-item similarity based on the sim function and data inputed
        if len(X)==0:
            self.sim = simfunction()            
        else:
            self.sim = simfunction(X[0]) # *X passes in a tuple format of (X,), to X[0] will be the actual transformed matrix
            
    def cossim(self):    
        """
        Calculates item-item similarity for all pairs of items using cosine similarity (values from 0 to 1) on utility matrix
        Returns a cosine similarity matrix of size (#all movies, #all movies)
        """
        # Return a sim matrix by calculating item-item similarity for all pairs of items using Jaccard similarity
        # Cosine Similarity: C(A, B) = (A.B) / (||A||.||B||) 
        # your code here

        # get movie ratings array
        movie_ratings_matrix = self.Mr
        # get average movie ratings using all users ratings
        #avg_user_ratings = self.data.test.uID.apply(lambda x: movie_ratings_matrix[self.uid2idx[x]].sum() / (movie_ratings_matrix[self.uid2idx[x]] != 0).sum())
        avg_movie_ratings_all_users = movie_ratings_matrix.sum(axis=1) / (movie_ratings_matrix != 0).sum(axis=1)
        # create a sparse matrix for operating cosine on its values
        movie_ratings_array = np.repeat(np.expand_dims(avg_movie_ratings_all_users, axis=1), movie_ratings_matrix.shape[1], axis=1)
        # take care of all the zero ratings
        movie_ratings_zeros = (movie_ratings_matrix==0)
        movie_ratings_array_adj = movie_ratings_matrix + movie_ratings_zeros * movie_ratings_array - movie_ratings_array
        # average all the ratings: divide by its magnitude
        mr_avg = movie_ratings_array_adj/np.sqrt((movie_ratings_array_adj**2).sum(axis=0))
        # set nans to 0
        mr_avg[np.isnan(mr_avg)] = 0
        # item-item cosine similarity
        item_item_cosine_sim = np.dot(mr_avg.T, mr_avg)
        # set diagonals to 1
        for i in range(len(self.allmovies)):
            item_item_cosine_sim[i, i] = 1
        
        # normalize cosine formula
        #guessing here based on the slides
        norm_cosine_sim = 0.5 + 0.5*item_item_cosine_sim
        
        return norm_cosine_sim
    
    def jacsim(self,Xr):
        """
        Calculates item-item similarity for all pairs of items using jaccard similarity (values from 0 to 1)
        Xr is the transformed rating matrix.
        """    
        # Return a sim matrix by calculating item-item similarity for all pairs of items using Jaccard similarity
        # Jaccard Similarity: J(A, B) = |A∩B| / |A∪B| 
        # your code here
        
        n_items = Xr.shape[1]
        max_xr = int(Xr.max())
        if max_xr > 1:
            intersection = np.zeros((n_items,n_items)).astype(int)
            for i in range(1, max_xr+1):
                csr = csr_matrix((Xr==i).astype(int))
                intersection = intersection + np.array(csr.T.dot(csr).toarray()).astype(int)
            nz_inter = intersection
        else:
            # Convert Xr into a CSR format
            csr0 = csr_matrix((Xr>0).astype(int))
            # Take the dot product
            nz_inter = np.array(csr0.T.dot(csr0).toarray()).astype(int)   
        
        # Formula jaccard similarity:
        A = (Xr>0).astype(bool)
        rowsum = A.sum(axis=0)
        rsumtile = np.repeat(rowsum.reshape((n_items,1)),n_items,axis=1)   
        union = rsumtile.T + rsumtile - nz_inter
        
        # Perform the two boundary checks:-
        #  - since dividing by magnitude may produce inf, zeros, etc. Set nans to 0.
        union[np.isnan(union)] = 0
        
        jaccard_sim = nz_inter / union
        
        #  - Covariance/correlation values for np.dot([M.T, M]) matrix should have 
        #    diagonal set to 1.
        for i in range(n_items):
            jaccard_sim[i, i] = 1
        
        return jaccard_sim
    
    
class test_recommender_system(unittest.TestCase):
    def setUp(self):
        

        # Creating Sample test data
        MV_users = pd.read_csv('data/users.csv')
        MV_movies = pd.read_csv('data/movies.csv')
        train = pd.read_csv('data/train.csv')
        test = pd.read_csv('data/test.csv')
        
        Data = namedtuple('Data', ['users','movies','train','test'])
        self.data = Data(MV_users, MV_movies, train, test)
        
        np.random.seed(42)
        self.sample_train = train[:30000]
        self.sample_test = test[:30000]


        self.sample_MV_users = MV_users[(MV_users.uID.isin(self.sample_train.uID)) | (MV_users.uID.isin(self.sample_test.uID))]
        self.sample_MV_movies = MV_movies[(MV_movies.mID.isin(self.sample_train.mID)) | (MV_movies.mID.isin(self.sample_test.mID))]


        self.sample_data = Data(self.sample_MV_users, self.sample_MV_movies, self.sample_train, self.sample_test)
        
        self.RecSys = RecSys(self.sample_data)
        self.ContentBased = ContentBased(self.data)
        self.Collaborative = Collaborative(self.data)

    def test_predict_everything_to_3(self):
        # Sample tests predict_everything_to_3 in class RecSys

        sample_rs = self.RecSys
        sample_yp = sample_rs.predict_everything_to_3()
        print(sample_rs.rmse(sample_yp))
        assert sample_rs.rmse(sample_yp)==approx(1.2642784503423288, abs=1e-3), "Did you predict everything to 3 for the test data?"

    def test_predict_to_user_average(self):
        # Sample tests predict_to_user_average in the class RecSys
        sample_rs = self.RecSys
        sample_yp = sample_rs.predict_to_user_average()
        print(sample_rs.rmse(sample_yp))
        assert sample_rs.rmse(sample_yp)==approx(1.1429596846619763, abs=1e-3), "Check predict_to_user_average in the RecSys class. Did you predict to average rating for the user?" 

    def test_content_based(self):
        cb = self.ContentBased
        # tests calc_movie_feature_matrix in the class ContentBased 
        assert(cb.Mm.shape==(3883, 18))
        
    def test_content_based_calc_item_similarity(self):
        # Sample tests calc_item_item_similarity in ContentBased class 

        sample_cb = ContentBased(self.sample_data)
        sample_cb.calc_item_item_similarity() 

        print(np.trace(sample_cb.sim))
        print(sample_cb.sim[10:13,10:13])
        
        assert(sample_cb.sim.sum() > 0), "Check calc_item_item_similarity."
        assert(np.trace(sample_cb.sim) == 3152), "Check calc_item_item_similarity. What do you think np.trace(cb.sim) should be?"

    def test_content_based_calc_item_sim2(self):
        
        sample_cb = ContentBased(self.sample_data)
        sample_cb.calc_item_item_similarity()
        ans = np.array([[1, 0.25, 0.],[0.25, 1, 0.],[0., 0., 1]])
        for pred, true in zip(sample_cb.sim[10:13, 10:13], ans):
            assert approx(pred, 0.01) == true, "Check calc_item_item_similarity. Look at cb.sim"
            
        # for a, b in zip(sample_MV_users.uID, sample_MV_movies.mID):
        #     print(a, b, sample_cb.predict_from_sim(a,b))

        # Sample tests for predict_from_sim in RecSys class 
        assert(sample_cb.predict_from_sim(245,276)==approx(2.5128205128205128,abs=1e-2)), "Check predict_from_sim. Look at how you predicted a user rating on a movie given UserID and movieID."
        assert(sample_cb.predict_from_sim(2026,2436)==approx(2.785714285714286,abs=1e-2)), "Check predict_from_sim. Look at how you predicted a user rating on a movie given UserID and movieID."

    def test_recsys_predict(self):
        # Sample tests method predict in the RecSys class 
        sample_cb = ContentBased(self.sample_data)
        sample_cb.calc_item_item_similarity()
        
        sample_yp = sample_cb.predict()
        sample_rmse = sample_cb.rmse(sample_yp)
        print(sample_rmse)

        assert(sample_rmse==approx(1.1962537249116723, abs=1e-2)), "Check method predict in the RecSys class."
        
        # Hidden tests method predict in the RecSys class 
        cb = ContentBased(self.data)
        yp = cb.predict()
        rmse = cb.rmse(yp)
        print(rmse)
    
    def test_cossim(self):
        # Sample tests cossim method in the Collaborative class

        sample_cf = Collaborative(self.sample_data)
        sample_cf.calc_item_item_similarity(sample_cf.cossim)
        sample_yp = sample_cf.predict()
        sample_rmse = sample_cf.rmse(sample_yp)

        assert(np.trace(sample_cf.sim)==3152), "Check cossim method in the Collaborative class. What should np.trace(cf.sim) equal?"
        assert(sample_rmse==approx(1.1429596846619763, abs=5e-3)), "Check cossim method in the Collaborative class. rmse result is not as expected."
        assert(sample_cf.sim[0,:3]==approx([1., 0.5, 0.5],abs=1e-2)), "Check cossim method in the Collaborative class. cf.sim isn't giving the expected results."
        
        # Hidden tests cossim method in the Collaborative class

        cf = Collaborative(self.data)
        cf.calc_item_item_similarity(cf.cossim)
        yp = cf.predict()
        rmse = cf.rmse(yp)
        print(rmse)
        
    def test_jacsim_gtr_3(self):
        # test jacsim method in the Collaborative class
        cf = Collaborative(self.data)
        Xr = cf.Mr>=3
        t0=time.perf_counter()
        cf.calc_item_item_similarity(cf.jacsim,Xr)
        t1=time.perf_counter()
        time_sim = t1-t0
        print('jacsim > 3 similarity calculation time',time_sim)
        yp = cf.predict()
        rmse = cf.rmse(yp)
        print(rmse)
        assert(rmse<0.99)
        
    def test_jacsim_gtr_1(self):
        # test jacsim method with ratings >= 1 in the Collaborative class
        cf = Collaborative(self.data)
        Xr = cf.Mr>=1
        t0=time.perf_counter()
        cf.calc_item_item_similarity(cf.jacsim,Xr)
        t1=time.perf_counter()
        time_sim = t1-t0
        print('jacsim > 1 similarity calculation time',time_sim)
        yp = cf.predict()
        rmse = cf.rmse(yp)
        print(rmse)
        assert(rmse<1.0)
        
    def test_jacsim_no_transform(self):
        # test jacsim method with no transformation in the Collaborative class
        cf = Collaborative(self.data)
        Xr = cf.Mr.astype(int)
        t0=time.perf_counter()
        cf.calc_item_item_similarity(cf.jacsim,Xr)
        t1=time.perf_counter()
        time_sim = t1-t0
        print('jacsim no transform - similarity calculation time',time_sim)
        yp = cf.predict()
        rmse = cf.rmse(yp)
        print(rmse)
        assert(rmse<0.96)

if __name__ == "__main__":
    
    # print("test code to run a few things")
    # test_RecSys = test_recommender_system()
    # test_RecSys.setUp()
    # test_RecSys.test_predict_everything_to_3()
    # print("\n\ntesting predict_to_user_average")
    # test_RecSys.test_predict_to_user_average()
    # print("\n\ntesting content_based")
    # test_RecSys.test_content_based()
    # print("\n\ntesting content_based_calc_item_similarity")
    # test_RecSys.test_content_based_calc_item_similarity()
    # print("\n\ntesting 2nd content_based_calc_item_similarity and predict_from_sim")
    # test_RecSys.test_content_based_calc_item_sim2()
    # print("\n\ntesting recsys_predict")
    # test_RecSys.test_recsys_predict()
    # print("\n\ntesting cossim")
    # test_RecSys.test_cossim()
    # print("\n\ntesting jacsim with ratings >= 3")
    # test_RecSys.test_jacsim_gtr_3()
    # print("\n\ntesting jacsim with ratings >= 1")
    # test_RecSys.test_jacsim_gtr_1()
    # print("\n\ntesting jacsim with no transformation")
    # test_RecSys.test_jacsim_no_transform()
    
    unittest.main(argv=[''], verbosity=2, exit=False)   