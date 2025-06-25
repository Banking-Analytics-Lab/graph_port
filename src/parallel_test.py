import pandas as pd
import numpy as np

import dcor._fast_dcov_avl
import numpy as np
from timeit import timeit
import matplotlib.pyplot as plt


# ret_df = pd.read_excel('Datasets/midcap_vol_2002_7_filtered.xlsx')

# ret_df.set_index(['Date'],inplace=True)
# ret_df.replace([np.inf, -np.inf], np.nan,inplace=True)

# ret_df.fillna(0,inplace=True)


n_times = 100
n_samples = 1000
n_comps_list = [10, 50, 100]

naive_times = np.zeros(len(n_comps_list))
cpu_times = np.zeros(len(n_comps_list))
parallel_times = np.zeros(len(n_comps_list))

for i, n_comps in enumerate(n_comps_list):
    x = np.random.normal(size=(n_comps, n_samples))
    y = np.random.normal(size=(n_comps, n_samples))

    def naive():
        return dcor.rowwise(
            dcor.distance_covariance_sqr, x, y, rowwise_mode=dcor.RowwiseMode.NAIVE
        )

    def cpu():
        return dcor.rowwise(
            dcor.distance_covariance_sqr,
            x,
            y,
            compile_mode=dcor.CompileMode.COMPILE_CPU,
        )

    def parallel():
        return dcor.rowwise(
            dcor.distance_covariance_sqr,
            x,
            y,
            compile_mode=dcor.CompileMode.COMPILE_PARALLEL,
        )

    naive_times[i] = timeit(naive, number=n_times)
    cpu_times[i] = timeit(cpu, number=n_times)
    parallel_times[i] = timeit(parallel, number=n_times)
    # gpu_times[i] = timeit(gpu, number=n_times)

plt.title("Distance covariance performance comparison")
plt.xlabel("Number of computations of distance covariance")
plt.ylabel("Time (seconds)")
plt.plot(n_comps_list, naive_times, label="naive")
plt.plot(n_comps_list, cpu_times, label="cpu")
plt.plot(n_comps_list, parallel_times, label="parallel")
plt.legend()
plt.show()
plt.savefig("test_fig.png")
