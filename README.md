# multiisotonic
An interface for multidimensional isotonic regression consistent with scikit-learn.

In one dimension, points are completely ordered (i.e. every point is either greater than, less than, or equal to every other point). This case is handled by the sklearn.isotonic module. In the multidimensional case, a complete ordering of points is no longer generally possible, but points may still be partially ordered: If all of the coordinates of the first point are less than or equal to the coordinates of the second point, then the first point can be deemed less than or equal to the second. For example, in two dimensions, the points (1,3) and (2,2) are not ordered, but the point (1,3) is less than the point (2,4), which is less than the point (2,5). A multidimensional isotonic function is guaranteed to yield values that are nondecreasing when evaluated at a series of nondecreasing points; for example, f(1,3) <= f(2,4) <= f(2,5).

A multidimensional isotonic regression takes a set of multidimensional points X, with corresponding values y, and returns an isotonic function with minimum squared distance from y. Algorithmically, this turns out to be mappable to a network flow problem (see [Picard 1976](http://dx.doi.org/10.1287/mnsc.22.11.1268) or [Spouge, Wan, and Wilbur 2003](http://dx.doi.org/10.1023/A:1023901806339)). This procedure is sensitive only to feature ranks, not values.

This package requires [scikit-learn](https://github.com/scikit-learn/scikit-learn) and [igraph](https://github.com/igraph/python-igraph), and all sub-dependencies.
