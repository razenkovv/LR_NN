import numpy as np

import hamming

np.random.seed(2)

vals = [-1, 1]
number = 50
dimension = 10
nn = hamming.Hamming(vals, hamming.generate_unique_test_samples(number=number, dimension=dimension, vals=vals))

test_obj = np.random.choice(vals, dimension)
dist = nn.distances(test_obj)
res = nn.maxnet(dist, eps=0.01)
ind = [i for i, r in enumerate(res) if r > 0.0]
nearest_samples = nn.storage[ind]

print("vals: ", nn.vals)
print("\nstorage:")
for i, s in enumerate(nn.storage):
    print(i, ": ", s,)
print("\ntest_object: ", test_obj)
print("\nmaxnet res:", res)
print("\nindexes: ", ind)
print("\nnearest samples:\n", nearest_samples)
print("\ndifferences:\n", nearest_samples - test_obj)
