from transformers import pipeline
import numpy as np
import torch




class Small_Molecule_Embedder:
    """
        A simple wrapper class for the creation of small molecule embeddings from SMILES strings.

        Allows one to pass in a HuggingFace model name for compound embedding.
        The default model is ChemBERTa: https://arxiv.org/abs/2010.09885

    """
    def __init__(self,
                model:str = "seyonec/ChemBERTa-zinc-base-v1",
                readout: str = 'mean'
    ):
        
        """
        Class constructor. Initializes the embedder by downloading the HuggingFace Model.

        Also lets one specify how to summarize the full sequence. 
        """
        self.model =  pipeline(task="feature-extraction", model=model) 
        assert readout in ['mean','median','full'] ,"readout method must be one of ['mean','median','full']"
        self.readout = readout


    def embed_molecule(self,
                    mol_SMILES:str,
                    ):
            """
            Function to construct molecule embedding for a string and construct sequence-wise readouts. 

            """
        
            
            res = self.model(mol_SMILES,return_tensors=True)
    
            res = res.numpy().squeeze()
            if self.readout == 'mean':
                emb = np.mean(res,axis=0)
            elif self.readout == 'median':
                emb = np.median(res,axis=0)
            else:
                emb = res
            return emb
                 