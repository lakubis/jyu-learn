import numpy as np

# Here, we are going to see how numpy processes the values by evaluating arr1, arr2, and arr3.

arr1 = np.array([100, 1e-14])
arr2 = np.array([100, 1e-14, 100])
arr3 = np.array([100, 1e-14, 100, 1e-14, 100, 1e-14, -300])
print(np.sum(arr1)) # Output = 100.00000000000001
print(np.sum(arr2)) # Output = 200.0
print(np.sum(arr3)) # Output = 0.0

'''
Explanation:
Analytically, we should have gotten 3e-14 instead of 0.0.

Numerically, numpy process the numbers one-by-one. As we can see, the sum of arr1 is 100.00000000000001. It already exhausted the precision of the float.

As a result, when we add another 100 to the float, it loses information on the 1e-14. This is why the sum of arr2 is 200.0

This loss of information repeats, and we ended up with 300.0 before we finally substract 300.0, hence the sum of arr3 is 0.0
'''