import numpy as np


class FilteredComplex:
    """
    Implements a simplicial complex filtered by the real line.
    """
    def __init__(self, simplices, births):
        self._simplices = simplices
        self._births = births

    def cut(self, threshold):
        #assert threshold >= 0 and threshold <= 1
        return [
            simplices_of_dimension[births_of_dimension >= threshold]
            for simplices_of_dimension, births_of_dimension in zip(self._simplices, self._births)
        ]


    def to_simplex_tree(self, log_normalization=True):
        import gudhi
        simplex_tree = gudhi.SimplexTree()

        for simplices_of_dimension, births_of_dimension in zip(self._simplices, self._births):
            births = -births_of_dimension if not log_normalization else -np.log(births_of_dimension)
            simplex_tree.insert_batch(simplices_of_dimension.T, births)
        
        return simplex_tree
