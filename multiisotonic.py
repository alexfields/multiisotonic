# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 13:32:58 2014

@author: afields
"""

import numpy as np
from scipy import sparse
import igraph
import multiprocessing as mp
from sklearn.base import BaseEstimator, RegressorMixin


class MultiIsotonicRegressor(BaseEstimator, RegressorMixin):
    """Regress a target value as a non-decreasing function of each input attribute,
    when the other attributes are non-decreasing

    min_partition_size is the minimum allowable size to which to partition the
    training set, to avoid overfitting
    """
    def __init__(self, min_partition_size=1):
        self.min_partition_size = min_partition_size

    def fit(self, X, y, njob=1):
        """Fit a multidimensional isotonic regression model

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Training data

        y : array-like, shape=(n_samples,)
            Target values

        Returns
        -------
        self : object
            Returns an instance of self
        """
        assert len(X) == len(y)
        X = np.array(X)

        myorder = np.argsort(X[:, 0])  # order along the first axis to at least avoid some of the comparisons
        self._training_set = X[myorder, :]
        ysort = np.array(y, dtype=np.float64)[myorder]

        indices = []
        indptr = [0]
        for (i, Xrow) in enumerate(self._training_set[:, 1:]):
            indices.append(np.flatnonzero((Xrow <= self._training_set[i+1:, 1:]).all(1))+i+1)
            indptr.append(indptr[-1]+len(indices[-1]))
        all_comparisons = sparse.csr_matrix((np.ones(indptr[-1], dtype=np.bool), np.concatenate(indices), indptr),
                                            shape=(X.shape[0], X.shape[0]), dtype=np.bool)
        edges_to_add = zip(*(all_comparisons-all_comparisons.dot(all_comparisons)).nonzero())
        mygraph = igraph.Graph(n=len(y), edges=edges_to_add, directed=True, vertex_attrs={'y': ysort})

        def _add_source_sink(graph_part):
            """ Add in the edges connecting the source and sink vertices to the internal nodes of the graph
            """
            y_part = np.array(graph_part.vs['y'])
            y_part -= y_part.mean()
            maxval = np.abs(y_part).sum()+1
            vsrc = graph_part.vcount()
            vsnk = vsrc+1
            graph_part.add_vertices(2)
            src_snk_edges = [(vsrc, curr_v) if curr_y > 0 else (curr_v, vsnk) for (curr_v, curr_y) in enumerate(y_part)]
            n_internal_edges = graph_part.ecount()
            graph_part.add_edges(src_snk_edges)
            graph_part.es['c'] = ([maxval]*n_internal_edges)+list(np.abs(y_part))

        def _partition_graph(inQ, outQ):
            """
            inQ is a mp.JoinableQueue into which lists of vertices to partition are dumped
            outQ is a mp.Queue into which the final partitions are dumped
            """
            while True:
                origV = inQ.get(True)
                if origV is None:  # poison pill
                    inQ.put(None)
                    inQ.task_done()
                    return
                currgraph = mygraph.subgraph(origV)
                _add_source_sink(currgraph)
                currpart = currgraph.mincut(currgraph.vcount()-2, currgraph.vcount()-1, 'c').partition
                if len(currpart[0])-1 < self.min_partition_size or len(currpart[1])-1 < self.min_partition_size:
                    # this partitioning would result in one of the sets being too small - so don't do it!
                    outQ.put(origV)
                else:
                    inQ.put([origV[idx] for idx in currpart[0][:-1]]) # keep partitioning
                    inQ.put([origV[idx] for idx in currpart[1][:-1]]) # keep partitioning
                inQ.task_done()

        graph_queue = mp.JoinableQueue()
        partition_queue = mp.Queue()
        workers = [mp.Process(target=_partition_graph, args=(graph_queue, partition_queue)) for _ in xrange(njob)]
        for worker in workers:
            worker.start()
        graph_queue.put(range(len(y)))
        graph_queue.join()
        graph_queue.put(None)

        nodes_to_cover = len(y)
        self._training_set_scores = np.empty(len(y))
        while not partition_queue.empty():
            part = partition_queue.get()
            self._training_set_scores[part] = ysort[part].mean()
            nodes_to_cover -= len(part)
        assert nodes_to_cover == 0

    def predict(self, X):
        """Predict according to the isotonic fit

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)

        Returns
        -------
        C : array, shape=(n_samples,)
            Predicted values
        """
        res = np.zeros(len(X))
        for (i, Xrow) in enumerate(X):
            lower_training_set = (self._training_set <= Xrow).all(1)
            if lower_training_set.any():  # if below the entire training set, leave it at zero!
                res[i] = self._training_set_scores[lower_training_set].max()
        return res
