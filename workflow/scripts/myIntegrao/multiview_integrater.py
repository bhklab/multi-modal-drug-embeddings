from typing import List, Tuple, Union, Optional, Dict
import pandas as pd
import numpy as np 
from collections import defaultdict

class multi_view_integrater:
    def __init__(
            self,
            views: List[pd.DataFrame],
            view_names:List[str] = None,
            metrics: List[str] = None,
            neighbor_size=None,
            mu: float = 0.5,
            norm_factor: float = 1.0,
            random_seed:int = 42,
            num_fusion_iters:int = 20,
    ):
        

        """
        Initializes an object that integrates mutliple 'views' of a kind of objects
        


        Parameters
        ----------

        views : List[pd.Dataframe]
                    List of views (datasets) for consideration. 
                    
                    These are expected to be pandas dataframes where each row corresponds to a sample
                    and each column to a feature. The datasets can have different numbers of samples 
                    as well as different feature dimensions. Samples in a single view should have the
                    same number of features. 

        view_names: List[str], default None
                    List of the names for each view. These are optional but useful.


        metrics: List[str], default None
                    List of metrics to use when computing affinity for each view. Default is 'None' which uses
                    the square Euclidean distance for each. 

        
        Returns
        --------

        None

    

        """
        
        assert len(views)>=1, "must have at least one view"
        
        self.views = views
        self.view_names = view_names
        
        if metrics is None:
            self.metrics = ['sqeuclidean']*len(self.views)
        else:
            assert len(metrics) == len(views), "Number of metrics and number of views must be the same or metrics must be None"
            self.metrics = metrics

    

    def _make_affinities(self
                         )->None:
        pass
        
    def _index_views(self)-> None:
        
        """
        Index the views for mutli-view fusion.

        
        Returns
        -------
        matrices_pure: Expression matrices without the first column and first row
        dict_commonSample: dictionaries that give you the common samples between 2 views
        dict_uniqueSample: dictionaries that give you the unique samples between 2 views
        original_order: the original order of samples for each view
        """

        self.original_order = [0] * (len(self.views))
        self.dict_original_order = {}
        self.dict_commonSample = {}
        self.dict_uniqueSample = {}
        self.dict_commonSampleIndex = {}
        self.dict_sampleToIndexs = defaultdict(list)

        for i in range(0, len(self.views)):
            self.original_order[i] = list(self.views[i].index)
            self.dict_original_order[i] = self.original_order[i]
            for sample in self.original_order[i]:
                self.dict_sampleToIndexs[sample].append(
                    (i, np.argwhere(self.views[i].index == sample).squeeze().tolist())
                )

        for i in range(0, len(self.original_order)):
            for j in range(i + 1, len(self.original_order)):
                commonList = list(set(self.original_order[i]).intersection(self.original_order[j]))
                print("Common sample between view{} and view{}: {}".format(i, j, len(commonList)))
                self.dict_commonSample.update(
                    self.dict_commonSample.fromkeys([(i, j), (j, i)], commonList)
                )
                self.dict_commonSampleIndex[(i, j)] = [
                    np.argwhere(self.views[i].index == x).squeeze().tolist()
                    for x in commonList
                ]
                self.dict_commonSampleIndex[(j, i)] = [
                    np.argwhere(self.views[j].index == x).squeeze().tolist()
                    for x in commonList
                ]

                self.dict_uniqueSample[(i, j)] = list(
                    set(self.original_order[i]).symmetric_difference(commonList)
                )
                self.dict_uniqueSample[(j, i)] = list(
                    set(self.original_order[j]).symmetric_difference(commonList)
                )

        
            
    

        def