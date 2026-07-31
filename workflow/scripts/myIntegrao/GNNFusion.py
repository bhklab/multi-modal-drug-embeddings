from snf.compute import _flatten, _B0_normalized, _find_dominate_set

from typing import List, Tuple, Union, Optional, Dict
import pandas as pd
import numpy as np
from integrao.main import dist2, integrao_fuse, _stable_normalized

def _scaling_normalized_pd(W, ratio):
    """
    Adds `alpha` to the diagonal of pandas dataframe `W`

    Parameters
    ----------
    W : (N, N) array_like
        Similarity array from SNF

    Returns
    -------
    W : (N, N) np.ndarray
        Stable-normalized similiarity array
    """

    # add `alpha` to the diagonal and symmetrize `W`
    rowSum = np.sum(W, 1) - np.diag(W)
    rowSum[rowSum == 0] = 1

    W = (W / rowSum) * 0.5 * ratio 

    W_np = W.values
    np.fill_diagonal(W_np, 1-0.5*ratio)
    W = pd.DataFrame(W_np, index=W.index, columns=W.columns) 

    W = check_symmetric(W, raise_warning=False)

    return W

def _stable_normalized(W):
    """
    Adds `alpha` to the diagonal of pandas dataframe `W`

    Parameters
    ----------
    W : (N, N) array_like
        Similarity array from SNF

    Returns
    -------
    W : (N, N) np.ndarray
        Stable-normalized similiarity array
    """

    # add `alpha` to the diagonal and symmetrize `W`
    rowSum = np.sum(W, 1) - np.diag(W)
    rowSum[rowSum == 0] = 1

    W = W / (2 * rowSum)

    W_np = W.values
    np.fill_diagonal(W_np, 0.5)
    W = pd.DataFrame(W_np, index=W.index, columns=W.columns) 

    W = check_symmetric(W, raise_warning=False)

    return W


def _multi_view_fusion(
        affinities: List[pd.DataFrame],
        sample_overlaps: Dict[Tuple[int,int],List[str]],
        unique_samples: Dict[Tuple[int,int],List[str]],
        original_orders:List[List[str]],
        num_nbrs:int = 20,
        num_iters:int = 20,
        norm_factor:float = 1.0,
        alpha: float = 0.2
    ) -> List[pd.DataFrame]:
    
    """
    Args:
        affinities: List of square affinity matrices as Pandas dataframes. They must all be square but
                    they do not need to be all of the same size
        
        sample_overlaps: A dictionary mapping pairs of indices to overlapping samples of views. It is symmetric, so 
                        sample_overlaps[(0,1)] = sample[(1,0)] = <List of Sample Names>

        unique_samples: A dictionary with set subtractions
                unique_samples[(0,1)] := meaning the unique samples from view 1 that are not in view 2

        original_orders: List of samples in the original order provided

        num_nbrs: int the number of fusion iterations to be performed

        num_iters: int the number of fusion iterations. Needs to be at least 1

        norm_factor: float  used to normalize the network 


    Returns:
        Ws: List of square pd Dataframes
    """


    Ws = [0] * len(affinities) # Initialize the Ws as the 0

    for n, mat in enumerate(affinities):
        
        # normalize affinity matrix based on strength of edges
        # mat = mat / np.nansum(mat, axis=1, keepdims=True) 
        affinities[n] = _stable_normalized_pd(mat)
        # aff[n] = check_symmetric(mat, raise_warning=False)

        # apply KNN threshold to normalized affinity matrix
        # We need to crop the intersecting samples from newW matrices
        neighbor_size = min(int(neighbor_size), mat.shape[0])
        Ws[n] = _find_dominate_set(aff[n], neighbor_size)



    for j in range(num_iters):
        pass

        aff_next = []
        for k in range(len(aff)):
            aff_temp = aff[k].copy()
            for col in aff_temp.columns:
                aff_temp[col].values[:] = 0
            aff_next.append(aff_temp)

        
        for i, view1 in enumerate(aff):
            # temporarily convert nans to 0 to avoid propagation errors
            nzW = newW[n]  # TODO: not sure this is a deep copy or not

            for j, view2 in enumerate(aff):
                if n == j:
                    continue

             # reorder mat_tofuse to have the common samples

             # why include the unique ones?
                mat_tofuse = mat_tofuse.reindex(
                    (sorted(dicts_common[(j, n)]) + sorted(dicts_unique[(j, n)])),
                    axis=1,
                )
                mat_tofuse = mat_tofuse.reindex(
                    (sorted(dicts_common[(j, n)]) + sorted(dicts_unique[(j, n)])),
                    axis=0,
                )

                # Next, let's crop mat_tofuse
                num_common = len(dicts_common[(n, j)])
                to_drop_mat = mat_tofuse.columns[
                    num_common : mat_tofuse.shape[1]
                ].values.tolist()
                mat_tofuse_crop = mat_tofuse.drop(to_drop_mat, axis=1)
                mat_tofuse_crop = mat_tofuse_crop.drop(to_drop_mat, axis=0)

                # Next, add the similarity from the view to fused to the current view identity matrix
                nzW_identity = pd.DataFrame(
                    data=np.identity(nzW.shape[0]),
                    index=original_order[n],
                    columns=original_order[n],
                )


                mat_tofuse_union = nzW_identity + mat_tofuse_crop
                mat_tofuse_union.fillna(0.0, inplace=True)
                mat_tofuse_union = _scaling_normalized_pd(mat_tofuse_union, ratio=mat_tofuse_crop.shape[0]/nzW_identity.shape[0])
                mat_tofuse_union = check_symmetric(mat_tofuse_union, raise_warning=False)
                mat_tofuse_union = mat_tofuse_union.reindex(original_order[n], axis=1)
                mat_tofuse_union = mat_tofuse_union.reindex(original_order[n], axis=0)

                # Now we are ready to do the diffusion
                nzW_T = np.transpose(nzW)
                aff0_temp = nzW.dot(
                    mat_tofuse_union.dot(nzW_T)
                )  # Matmul is not working, but .dot() is good


                ################################################# 
                # Experimentally introduce a weighting machanisim, use the exponential weight; Already proved it's not a good idea
                # num_com = mat_tofuse_crop.shape[0] / aff[n].shape[0]
                # alpha = pow(2, num_com) - 1
                # aff0_temp = alpha * aff0_temp + (1-alpha) * aff[n]

                #aff0_temp = _B0_normalized(aff0_temp, alpha=normalization_factor)
                aff0_temp = _stable_normalized_pd(aff0_temp)
                # aff0_temp = check_symmetric(aff0_temp, raise_warning=False)

                aff_next[n] = np.add(aff0_temp, aff_next[n])

            aff_next[n] = np.divide(aff_next[n], len(aff) - 1)
            # aff_next[n] = _stable_normalized_pd(aff_next[n])


    
