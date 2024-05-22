import numpy as np

l, m, n = 3, 4, 5
array = np.ones((l, m, n), dtype=int)

print(array.shape)
# Output: (3, 4, 5)

# Accessing the first element
print(array[0, 0, 0])
# Output: 1

# Accessing the last element
print(array[2, 3, 4])
# Output: 1
print(array[3, 4, 5])
# Output: 1