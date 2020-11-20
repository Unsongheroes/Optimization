import numpy as np
from sklearn.cluster import KMeans
import random as random

class NearstSubClassCentroidClassifer:
    def fit(self ,data, labels, nr_cluster):
        model = [{'category': category, 'c': cluster} for category in set(labels) for cluster in self.getClustersForCategoryPoints(data, labels, category, nr_cluster)]
        return model

    def getClustersForCategoryPoints(self, data, labels, category, nr_cluster):
        #Find labels in category
        
        new_label = [index for index, value in enumerate(labels) if value == category]
        #Find points in category
        categoryPoints = [data[index] for index in new_label]
        #Flatten points
        flattenPoints = [point.flatten() for point in categoryPoints]
        clusters = self.findClusters( flattenPoints, nr_cluster)
        return clusters

    def findClusters(self, data, nr_cluster):
        """ kmeans = KMeans(nr_cluster)
        kmeans.fit(data)
        c_kmeans = kmeans.predict(data)
        return c_kmeans """
        clusters = [random.choice(data) for _ in range(nr_cluster)]
        
        for _ in range(10):
            cluster_points = self.assign_vector_to_cluters(data, clusters)
             #update the cluster mean vectors
            clusters = self.calculate_mean_points( cluster_points)
        return clusters

    def calculate_mean_points(self, datalist):
        return [np.mean(cluster, axis=0) for cluster in datalist]


    def assign_vector_to_cluters(self, data, clusters): 
        distance_to_clusters = [self._get_closest_cluster_index(point, clusters,index) for index,point in enumerate(data)]
        clusters_points = [[] for _ in clusters]
        for index, point in enumerate(data):
            cluster_index = distance_to_clusters[index]
            clusters_points[cluster_index].append(point)
        return clusters_points

    def _get_closest_cluster_index(self, point, clusters, index):
        #Find distance for each cluster
        distance_to_clusters = [self.distance(cluster, point) for cluster in clusters]
        
        
        #find closest cluster index
        closest_cluster_index = distance_to_clusters.index(min(distance_to_clusters))
        
        return closest_cluster_index

    def distance(self, cluster, data):
        val = np.linalg.norm(cluster - data)
        return val

    def findcategories(self, model, data):
        #Calculate distance to each category
        distances = [np.linalg.norm(category['c'] - data.flatten()) for category in model]
        #Find index of minimum distance
        model_index = distances.index(min(distances))
        #Get category corrosponding to minimum distance
        category = model[model_index]['category']
        #Return closest category
        return category
        
    def score(self, model, data, labels):
        positives = [True for index, point in enumerate(data) if self.findcategories(model, point) == labels[index]]
        accuracy = len(positives)/len(data)
        return accuracy
