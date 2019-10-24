"""
Check the numpy / scipy methods we use.

Mostly the question at the moment is
how to get numerically stable `from_product`,
i.e. which matrix inverse function to use,
or how to rewrite `from_product` in a better way.
"""
import numpy as np

from multinorm import MultiNorm


mean = np.array([1e-20, 1, 1e20])
err = 1 * mean
names = ["a", "b", "c"]
mn = MultiNorm.from_error(mean, err, names=names)

print(mn)
print(mn.cov.values)

# BAD
try:
    print(mn.precision.values)
except np.linalg.LinAlgError:
    print("SINGULAR m3.precision")

# GOOD
print(np.linalg.inv(mn.cov.values))
