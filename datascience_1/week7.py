from week6 import *

def row_to_vect(row, features):
    vect = []
    for feature in features:
        vect.append(float(row[feature]))
    return tuple(vect)

def initialize_centroids(sample_table, features):
    k = len(sample_table)
    centroids = {}
    for i in range(k):
        row = sample_table.iloc[i]
        vector = row_to_vect(row, features)
        centroids[i] = {'centroid': vector, 'cluster': []}
        
    return centroids

def euclidean_distance(vect1, vect2):
    sum = 0
    for i in range(len(vect1)):
        sum += (vect1[i] - vect2[i])**2
    return sum**.5  # I claim that this square root is not needed in K-means - see why?

def closest_centroid(centroids, row, k):
    min_distance = euclidean_distance(centroids[0]['centroid'], row)
    min_centroid = 0
    for i in range(1,k):
        distance = euclidean_distance(centroids[i]['centroid'], row)
        if distance < min_distance:
            min_distance = distance
            min_centroid = i
    return (min_centroid, min_distance)

def phase_1(centroids, table, features, k):
    for i in range(k):
        centroids[i]['cluster'] = []  # starting new phase 1 so empty out values from prior iteration
    
    #Go through every row in Titanic table (or Loan table) and place in closest centroid cluster.
    for i in range(len(table)):
        row = table.iloc[i]
        vrow = row_to_vect(row, features)
        (index, dist) = closest_centroid(centroids, vrow, k)
        centroids[index]['cluster'].append(vrow)
    
    return centroids

#cluster is a list of points, i.e., a list of lists.
def compute_mean(cluster):
    if len(cluster) == 0:
        return []
    the_sum = cluster[0]  # use 0th point as starter
    
    #I am using zip to pair up all points then do addition
    for i in range(1,len(cluster)):
        the_sum = [pair[0]+pair[1] for pair in zip(the_sum, cluster[i])]
    n = len(cluster)*1.0
    the_mean_point = [x/n for x in the_sum]
    return the_mean_point


def phase_2(centroids, k, threshold):
    old_centroids = []
    
    stop = True
    #Compute k new centroids and check for stopping condition
    for i in range(k):
        current_centroid = centroids[i]['centroid']
        new_centroid = compute_mean(centroids[i]['cluster'])
        centroids[i]['centroid'] = new_centroid
        if euclidean_distance(current_centroid, new_centroid) > threshold:
            stop = False  # all it takes is one

    return (stop, centroids)


def k_means(table, features, k, hypers):
    n = 100 if 'n' not in hypers else hypers['n']
    threshold = 0.0 if 'threshold' not in hypers else hypers['threshold']
    
    centroid_table = table.sample(n=k, replace=False, random_state=100)  # only random choice I am making
    centroids = initialize_centroids(centroid_table, features)
    
    j = 0
    stop = False
    while( j < n and not stop):
        print('starting '+str(j+1))
        centroids = phase_1(centroids, table, features, k)
        (stop, centroids) = phase_2(centroids, k, threshold)
        j += 1
    print('done')
    return centroids

