from typing import Dict

def unpack_parameters(
    D:Dict,
    return_dict:bool=False
    ):
	
	if len(D.values())>1:
		return D if return_dict else tuple(D.values())
	else:
		return D if return_dict else tuple(D.values())[0]

