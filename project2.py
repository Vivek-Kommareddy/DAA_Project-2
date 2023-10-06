def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0

        while i < len(left_half) and j < len(right_half):
            if left_half[i][1] > right_half[j][1]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

def compute_pareto_optimal(points):
    # sort the points in descending order of Y values using merge sort
    merge_sort(points)

    # the first point is always Pareto-optimal
    pareto_optimal_points = [points[0]]

    max_x = points[0][0]

    for i in range(1, len(points)):
        if points[i][0] > max_x:
            pareto_optimal_points.append(points[i])
            max_x = points[i][0]

    return pareto_optimal_points

# # Example usage:
# points = [(1, 2), (4, 6), (2, 3), (5, 1), (3, 5), (7, 4)]
# pareto_optimal = compute_pareto_optimal(points)
# print(pareto_optimal)

# generate random points
import random
import time

random.seed(time.time())

n_values = []
times = []
for n in [10, 100, 1000, 10000, 100000, 1000000]:
    points = [(random.randint(1, 100000), random.randint(1, 100000)) for _ in range(n)]
    t0 = time.time()
    pareto_optimal = compute_pareto_optimal(points)
    t1 = time.time()
    print(n, len(pareto_optimal), t1 - t0)
    times.append(t1 - t0)
    n_values.append(n)

# # Plot the points
# plt.scatter([point[0] for point in points], [point[1] for point in points])

# # Plot the Pareto-optimal points
# plt.scatter([point[0] for point in pareto_optimal], [point[1] for point in pareto_optimal], color='red', marker='x')


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

nlogn=[n * np.log2(n) for n in n_values]

# average experimental time / average theoretical time
scaling_factor = np.mean(times) / np.mean(nlogn)

print('Scaling factor:', scaling_factor)

adj_nlogn = [scaling_factor * n * np.log2(n) for n in n_values]

# save to csv
df=pd.DataFrame(columns=['n', 'experimental', 'theoretical', 'adj_theoretical'])
df['n']=n_values
df['experimental']=times
df['theoretical']=nlogn
df['adj_theoretical']=adj_nlogn
df.to_csv('pareto.csv', index=False)


# plot
plt.plot(n_values, adj_nlogn, label='Adjusted theoretical')
plt.plot(n_values, times, label='Experimental')
plt.xlabel('Number of points')
plt.ylabel('Time taken (seconds)')
plt.xscale('log')
plt.title('Time complexity of the algorithm')
plt.legend()
plt.show()


