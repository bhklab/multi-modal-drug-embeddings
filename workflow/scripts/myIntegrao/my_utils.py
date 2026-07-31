import pandas as pd

def rescale_to_rw_matrix(
        W:pd.DataFrame,
        alpha:float =0.5):
    """
    
    Rescales the initial affinity matrices to being random walk matrices (P[i,j] in the paper. 

    The random walk stays put at node j with probabiltiy alpha and leaves with 
    Parameters
    ----------
    W : (N, N) array_like
        Similarity array from SNF

    Returns
    -------
    W : (N, N) np.ndarray
        Stable-normalized similiarity array
    """

    assert 0<=alpha<=1, "alpha must be in [0,1]"
    # add `alpha` to the diagonal and symmetrize `W`
    rowSum = np.sum(W, 1) - np.diag(W)
    rowSum[rowSum == 0] = 1

    W = W / (2 * rowSum)

    W_np = W.values
    np.fill_diagonal(W_np, alpha)
    W = pd.DataFrame(W_np, index=W.index, columns=W.columns) 

    W = check_symmetric(W, raise_warning=False)

    return W

