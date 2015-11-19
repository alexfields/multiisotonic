# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 13:32:58 2014

@author: afields
"""

import numpy as np
from scipy import sparse
import igraph
import multiprocessing as mp


class monotonic_classifier(object):
    """Evaluate probability of being in a particular class under the assumption
    that this probability is a non-decreasing function of each input attribute,
    when the other attributes are non-decreasing

    min_partition_size is the minimum allowable size to which to partition the
    training set, to avoid overfitting
    """
    def __init__(self, min_partition_size=1):
        self.min_partition_size = min_partition_size

    def fit(self, X, y, njob=1, verbose=False):
        """ Train the classifier on a training set

        X is an array of shape (M,N), with N features on M samples and with a partial order

        y is an array of shape (M,), giving the class of each sample
        A typical training set might denote positive examples as +1 and negatives as -1

        The monotonicity assumption asserts that for points A and B in feature space,
        if each of B's features is greater than or equal to the corresponding feature
        of A, then B's probability of being in the greater-value class is at least
        as high as A's probability.
        """
        assert len(X) == len(y)
        target_classes = np.unique(y)
        assert target_classes.size == 2  # larger numbers of classes is not implemented
        X = np.array(X)

        myorder = np.argsort(X[:, 0])  # order along the first axis to at least avoid some of the comparisons
        self._training_set = X[myorder, :]
        ysort = np.array(y, dtype=np.float64)[myorder]
        ysort[ysort == target_classes[0]] = 0.
        ysort[ysort == target_classes[1]] = 1.
#        maxval = np.abs(ysort-ysort.mean()).sum()+1

#        all_comparisons = np.zeros((len(y),len(y)),dtype=np.bool) # (i,j) element is True iff (X[i,:]<=X[j,:]).all(1)
#        for (i,Xrow) in enumerate(self._training_set[:,1:]): # already guaranteed to be increasing along the 0th column
#            all_comparisons[i,i:] = (Xrow<=self._training_set[i:,1:]).all(1)
#        edges_to_add = []
#        for (i,yval) in enumerate(ysort):
#            nodes_above_todo = all_comparisons[i,i+1:]
#            while nodes_above_todo.any():
#                j = np.flatnonzero(nodes_above_todo)[0]+i+1
#                edges_to_add.append((i,j))
#                nodes_above_todo &= ~all_comparisons[j,i+1:]
        indices = []
        indptr = [0]
        for (i, Xrow) in enumerate(self._training_set[:, 1:]):
            indices.append(np.flatnonzero((Xrow <= self._training_set[i+1:, 1:]).all(1))+i+1)
            indptr.append(indptr[-1]+len(indices[-1]))
        all_comparisons = sparse.csr_matrix((np.ones(indptr[-1], dtype=np.bool), np.concatenate(indices), indptr),
                                            shape=(X.shape[0], X.shape[0]), dtype=np.bool)
        edges_to_add = zip(*(all_comparisons-all_comparisons.dot(all_comparisons)).nonzero())
        mygraph = igraph.Graph(n=len(y), edges=edges_to_add, directed=True,
                               vertex_attrs={'y': ysort})  # 'idx':range(len(y)),
#                               edge_attrs={'c':[maxval]*len(edges_to_add)})
#        _add_source_sink(mygraph) # source and sink get added during partitioning

        def _add_source_sink(graph_part):
            """ Add in the edges connecting the source and sink vertices to the internal nodes of the graph
            """
#            y_part = ysort[graph_part.vs['idx']]
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
#            for (curr_v,curr_y) in enumerate(y_part):
#                if curr_y > 0:
#                    graph_part.add_edge(vsrc,curr_v,c=curr_y)
#                else:
#                    graph_part.add_edge(curr_v,vsnk,c=-curr_y)

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
                    inQ.put([origV[idx] for idx in currpart[0][:-1]])  # keep partitioning
                    inQ.put([origV[idx] for idx in currpart[1][:-1]])  # keep partitioning
#                if len(currpart[0]) == 1 or len(currpart[1]) == 1: #we're done
#                    outQ.put(origV)
#                else:
#                    if len(currpart[0])-1 <= self.max_partition_size:
##                        outQ.put(currgraph.vs['idx'][currpart[0][:-1]]) # small enough, so stop here
#                        outQ.put([origV[idx] for idx in currpart[0][:-1]]) # small enough, so stop here
#                    else:
##                        inQ.put(currgraph.vs['idx'][currpart[0][:-1]]) # keep partitioning
#                        inQ.put([origV[idx] for idx in currpart[0][:-1]]) # keep partitioning
#                    if len(currpart[1])-1 <= self.max_partition_size:
##                        outQ.put(currgraph.vs['idx'][currpart[1][:-1]]) # small enough, so stop here
#                        outQ.put([origV[idx] for idx in currpart[1][:-1]]) # small enough, so stop here
#                    else:
##                        inQ.put(currgraph.vs['idx'][currpart[1][:-1]]) # keep partitioning
#                        inQ.put([origV[idx] for idx in currpart[1][:-1]]) # keep partitioning
                inQ.task_done()

        graph_queue = mp.JoinableQueue()
        partition_queue = mp.Queue()
        workers = [mp.Process(target=_partition_graph, args=(graph_queue, partition_queue)) for i in xrange(njob)]
#        workers = mp.Pool(njob,_partition_graph,(graph_queue,partition_queue))
        for worker in workers:
            worker.start()
        graph_queue.put(range(len(y)))
        graph_queue.join()
        graph_queue.put(None)
#        for worker in workers:
#            worker.join()
#        workers.close()

        nodes_to_cover = len(y)
        self._training_set_scores = np.empty(len(y))
        while not partition_queue.empty():
            part = partition_queue.get()
            self._training_set_scores[part] = ysort[part].mean()
            nodes_to_cover -= len(part)
        assert nodes_to_cover == 0

    def predict_proba(self, X):
        """ Compute probabilities of a set of samples being in the positive class

        X is an array of shape (M,N), with N features on M samples

        Returns an array of shape (M,) of values between 0 and 1
        """
        proba_res = np.zeros(len(X))
        for (i,Xrow) in enumerate(X):
            lower_training_set = (self._training_set <= Xrow).all(1)
            if lower_training_set.any():  # if below the entire training set, leave it at zero!
                proba_res[i] = self._training_set_scores[lower_training_set].max()
        return proba_res


class monotonic_regressor(object):
    """Regress a target value as a non-decreasing function of each input attribute,
    when the other attributes are non-decreasing

    min_partition_size is the minimum allowable size to which to partition the
    training set, to avoid overfitting
    """
    def __init__(self, min_partition_size=1):
        self.min_partition_size = min_partition_size

    def fit(self, X, y, njob=1, verbose=False):
        """ Train the classifier on a training set

        X is an array of shape (M,N), with N features on M samples and with a partial order

        y is an array of shape (M,), giving the value of each sample

        The monotonicity assumption asserts that for points A and B in feature space,
        if each of B's features is greater than or equal to the corresponding feature
        of A, then B's probability of being in the greater-value class is at least
        as high as A's probability.
        """
        assert len(X) == len(y)
#        target_classes = np.unique(y)
#        assert target_classes.size == 2  # larger numbers of classes is not implemented
        X = np.array(X)

        myorder = np.argsort(X[:, 0])  # order along the first axis to at least avoid some of the comparisons
        self._training_set = X[myorder, :]
        ysort = np.array(y, dtype=np.float64)[myorder]
#        ysort[ysort==target_classes[0]] = 0.
#        ysort[ysort==target_classes[1]] = 1.

#        all_comparisons = np.zeros((len(y),len(y)),dtype=np.bool) # (i,j) element is True iff (X[i,:]<=X[j,:]).all(1)
#        for (i,Xrow) in enumerate(self._training_set[:,1:]): # already guaranteed to be increasing along the 0th column
#            all_comparisons[i,i:] = (Xrow<=self._training_set[i:,1:]).all(1)
#        edges_to_add = []
#        for (i,yval) in enumerate(ysort):
#            nodes_above_todo = all_comparisons[i,i+1:]
#            while nodes_above_todo.any():
#                j = np.flatnonzero(nodes_above_todo)[0]+i+1
#                edges_to_add.append((i,j))
#                nodes_above_todo &= ~all_comparisons[j,i+1:]
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
#            y_part = ysort[graph_part.vs['idx']]
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
#            for (curr_v,curr_y) in enumerate(y_part):
#                if curr_y > 0:
#                    graph_part.add_edge(vsrc,curr_v,c=curr_y)
#                else:
#                    graph_part.add_edge(curr_v,vsnk,c=-curr_y)

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
#                if len(currpart[0]) == 1 or len(currpart[1]) == 1: #we're done
#                    outQ.put(origV)
#                else:
#                    if len(currpart[0])-1 <= self.max_partition_size:
##                        outQ.put(currgraph.vs['idx'][currpart[0][:-1]]) # small enough, so stop here
#                        outQ.put([origV[idx] for idx in currpart[0][:-1]]) # small enough, so stop here
#                    else:
##                        inQ.put(currgraph.vs['idx'][currpart[0][:-1]]) # keep partitioning
#                        inQ.put([origV[idx] for idx in currpart[0][:-1]]) # keep partitioning
#                    if len(currpart[1])-1 <= self.max_partition_size:
##                        outQ.put(currgraph.vs['idx'][currpart[1][:-1]]) # small enough, so stop here
#                        outQ.put([origV[idx] for idx in currpart[1][:-1]]) # small enough, so stop here
#                    else:
##                        inQ.put(currgraph.vs['idx'][currpart[1][:-1]]) # keep partitioning
#                        inQ.put([origV[idx] for idx in currpart[1][:-1]]) # keep partitioning
                inQ.task_done()

        graph_queue = mp.JoinableQueue()
        partition_queue = mp.Queue()
        workers = [mp.Process(target=_partition_graph, args=(graph_queue, partition_queue)) for i in xrange(njob)]
#        workers = mp.Pool(njob,_partition_graph,(graph_queue,partition_queue))
        for worker in workers:
            worker.start()
        graph_queue.put(range(len(y)))
        graph_queue.join()
        graph_queue.put(None)
#        for worker in workers:
#            worker.join()
#        workers.close()

        nodes_to_cover = len(y)
        self._training_set_scores = np.empty(len(y))
        while not partition_queue.empty():
            part = partition_queue.get()
            self._training_set_scores[part] = ysort[part].mean()
            nodes_to_cover -= len(part)
        assert nodes_to_cover == 0

    def predict_proba(self, X):
        """ Compute probabilities of a set of samples being in the positive class

        X is an array of shape (M,N), with N features on M samples

        Returns an array of shape (M,) of values between 0 and 1
        """
        proba_res = np.zeros(len(X))
        for (i, Xrow) in enumerate(X):
            lower_training_set = (self._training_set <= Xrow).all(1)
            if lower_training_set.any():  # if below the entire training set, leave it at zero!
                proba_res[i] = self._training_set_scores[lower_training_set].max()
        return proba_res
