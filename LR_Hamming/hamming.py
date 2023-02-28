import numpy as np


class Hamming:
    def __init__(self, vals, storage):
        """
        :param storage: numpy array of initial samples
        """
        self.vals = vals
        self.storage = storage

    def distances(self, obj):
        """
        :param obj: object that should be compared to samples
        :return : distances between object and samples
        """
        res = self.storage/2 @ obj
        res[res < 0.0] = 0.0
        return res

    def maxnet(self, dist, eps=0.001):
        """
        :param dist: Hamming distances between obj and samples
        :param eps: epsilon
        :return dist: distances after stabilization
        """
        dist_new = np.zeros_like(dist)
        while np.sqrt(sum((dist_new - dist)**2)) > eps:
            for i in range(len(dist)):
                d = dist[i] - (sum(dist) - dist[i]) * eps
                dist_new[i] = d if d > 0.0 else 0.0
            dist, dist_new = dist_new, dist
        return dist


def generate_unique_test_samples(number, dimension, vals):
    if number > len(vals) ** dimension:
        raise Exception("Unique samples can't be generated")
    res = []
    while len(res) < number:
        res += np.random.choice(vals, (number, dimension)).tolist()
        res = np.unique(res, axis=0).tolist()
    return np.array(res[:number])
