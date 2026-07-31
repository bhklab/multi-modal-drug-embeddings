import yaml 
from damply import dirs
from utils import unpack_parameters


with open(dirs.CONFIG / "structure_embedding.yaml") as config_file:
	config = yaml.safe_load(config_file)
	
print(config)


layers, metrics, mu, nbhd_size, emb_dim = unpack_parameters(config['NETWORK_CONSTRUCTION'])
print(layers)