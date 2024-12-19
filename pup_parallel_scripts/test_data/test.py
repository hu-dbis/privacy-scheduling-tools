from bisect import bisect


def transform_to_weight(timestamp, min_t, max_t, scaling_factor=10):
    #return (timestamp - min_t) / (max_t - min_t) * (scaling_factor - 1) + 1
    bins = [int(min_t + i * (max_t - min_t) / scaling_factor) for i in range(1, scaling_factor)]
    print(bins)
    return scaling_factor - bisect(bins, timestamp) # highest weight assigned to lowest timestamps


# get distinct values from unix timestamps
# rank and assign to ranks to jobs as weights

# because probably very large number of timestamps, weight domain would be very large
# so we need to scale it down to a smaller range

if __name__ == '__main__':
    print(transform_to_weight(1614556800, 1614556800, 1614643200, 10))
    print(transform_to_weight(1614643200, 1614556800, 1614643200, 10))