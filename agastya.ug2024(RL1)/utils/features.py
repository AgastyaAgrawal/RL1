import abc 
import numpy as np 
from typing import Union, Literal 
class BaseExtractor(abc.ABC):

    @abc.abstractmethod # tells compiler that this is a blueprint. 
    def extract(self, state: np.ndarray) -> np.ndarray:
        # This converts a continuous state to a feature vector. 
        pass

    @property # remember that properties and actions have different call structures. And the following is a property of the feature vector. 
    @abc.abstractmethod
    def feature_dim(self) -> int:
        # Returns the dimensionality of the feature vector. 
        pass


#If I were to make a more sophisticated system, it would be better practice to not put in the np.ndarray constraint I suppose, 
# since people may prefer to use pytorch when using neural networks instead (better for autograd).

class TileCoder(BaseExtractor):
    def __init__(self, min_features: np.array, max_features: np.array, num_tiles: int, num_bins: int, displacement: np.array):
        self.min_features = min_features
        self.max_features = max_features
        self.num_tiles = num_tiles
        self.num_bins = num_bins
        self.bin_len = (max_features - min_features)/num_bins
        self.offset = displacement
    
    #we shall do a change of frame. Instead of moving the grid, we shall move the state iself. 
    def extract(self, state: np.ndarray) -> np.ndarray:
        num_features = state.shape[0]
        tiling_steps = np.arange(self.num_tiles)[:, None] #for broadcasting
        shifted_states = (state - self.min_features) + (tiling_steps*self.offset) #The first term is for shifting to 0. 
        bin_coords = np.floor(shifted_states/self.bin_len).astype(int)
        bin_coords = np.clip(bin_coords, 0, self.num_bins - 1)
        dim = tuple([self.num_bins]*num_features)
        flat_indices = np.ravel_multi_index(bin_coords.T, dim)
        tiling_block_size = self.num_bins ** num_features
        global_indices = flat_indices + (np.arange(self.num_tiles) * tiling_block_size)
        features = np.zeros(self.feature_dim)
        features[global_indices] = 1.0 #since we are using binary features, that is what it must return. 

        return features
    
    @property
    def feature_dim(self) -> int:
        num_features = len(self.min_features)
        return self.num_tiles * (self.num_bins ** num_features)

class RadialBasisFunctions(BaseExtractor):
    def __init__(self, min_features: np.ndarray, max_features: np.ndarray, num_centres: int,  norm: Union[int, Literal["inf"]], sigma: float = 1.0):
        #we add in the option of norm the user wants to chose depending on the situation, which can be an int or inf. Cases will be made for each. 
        self.min_features = min_features
        self.max_features = max_features
        self.norm = norm
        self.num_centres = num_centres
        self.sigma = sigma

        num_features = len(min_features)
        axes = [np.linspace(min_features[i], max_features[i], num_centres) for i in range(num_features)]
        grids = np.meshgrid(*axes, indexing='ij')
        self.centers = np.vstack([g.flatten() for g in grids]).T # we now have a list of centres 
    
    def extract(self, state: np.ndarray) -> np.ndarray:
        centred_vector = state - self.centers
        if (self.norm != "inf"):
            distances = np.linalg.norm(centred_vector, ord = self.norm, axis = 1)
            return np.exp(-(distances**self.norm)/(2*self.sigma**2))
        else:
            centred_vector = np.abs(centred_vector)
            distances = np.max(centred_vector, axis = 1)
            return np.exp(-(distances)/(2*self.sigma**2))
        
    @property
    def feature_dim(self) -> int:
        return self.num_centres**len(self.min_features)
    





