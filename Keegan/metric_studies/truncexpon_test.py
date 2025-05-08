import matplotlib.pyplot as plt

from scipy.stats import truncexpon

# truncexpon(b, loc=0, scale=1)
# samples from [loc, loc + b * scale]
a, b = 0.0, 1.0
lambda_ = 1.0

scale = 1.0 / lambda_
lower = a
upper = b
b_scaled = (upper - lower) / scale

rv = truncexpon(b=b_scaled, loc=lower, scale=scale)
samples = 1-rv.rvs(size=1000)

plt.hist(samples)
plt.savefig("trunc_test.pdf")
plt.clf()
