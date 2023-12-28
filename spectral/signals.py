import numpy as np
from functools import partial
import pywt 
from numpy.fft import fft2
import cv2
from typing import Any, Dict, Tuple, List, Optional

def create_grid(pd, resolution, global_max):
    """
    Creates a grid representation of persistence diagrams.
    
    Parameters:
    - pd (array): An array of persistence diagrams.
    - resolution (tuple): The resolution of the grid.
    - global_max (tuple): The maximum values for scaling the grid.
    
    Returns:
    - grid (numpy.ndarray): A grid representation of the input persistence diagrams.
    """
    grid = np.zeros(resolution)
    for birth, death in pd:
        x = int((birth / global_max[0]) * (resolution[0] - 1))
        y = int((death / global_max[1]) * (resolution[1] - 1))
        grid[x, y] += 1
    return grid

def wavelet_functional(pd, resolution, global_max, wave):
    """
    Applies a wavelet transform to the persistence diagram and calculates its total energy.
    
    Parameters:
    - pd (array): An array of persistence diagrams.
    - resolution (tuple): The resolution of the grid.
    - global_max (tuple): The maximum values for scaling the grid.
    - wave (str): The wavelet type.
    
    Returns:
    - (numpy.ndarray, float): A tuple containing the flattened wavelet coefficients and the total energy.
    """
    discretized_pd = create_grid(pd, resolution, global_max) 
    cA, (cH, cV, cD) = pywt.dwt2(discretized_pd, wave)
    total_energy = np.sum(np.square(cA))+np.sum(np.square(cH))+np.sum(np.square(cV))+np.sum(np.square(cD))
    return np.concatenate([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()]), total_energy


def fourier_functional(pd, resolution, global_max):
    """
    Applies a Fourier transform to the persistence diagram and calculates its magnitude, phase, and total energy.
    
    Parameters:
    - pd (array): An array of persistence diagrams.
    - resolution (tuple): The resolution of the grid.
    - global_max (tuple): The maximum values for scaling the grid.
    
    Returns:
    - (numpy.ndarray, float): A tuple containing concatenated magnitude and phase arrays, and the total energy.
    """
    discretized_pd = create_grid(pd, resolution, global_max)
    fft_output = fft2(discretized_pd)
    magnitude = np.abs(fft_output).flatten() 
    phase = np.angle(fft_output).flatten()
    total_energy = np.sum(np.square(magnitude))
    return np.concatenate([magnitude, phase]), total_energy


def gabor_functional(pd, resolution, global_max, sigma, lambd, gamma, theta, psi):
    """
    Applies a Gabor filter to the persistence diagram.
    
    Parameters:
    - pd (array): An array of persistence diagrams.
    - resolution (tuple): The resolution of the grid.
    - global_max (tuple): The maximum values for scaling the grid.
    - sigma, lambd, gamma, theta, psi (float): Gabor filter parameters.
    
    Returns:
    - (numpy.ndarray, int): A tuple containing the flattened output of the Gabor filter and a constant value 0.
    """
    kernel_x = int(resolution[0]/2)
    kernel_y = int(resolution[1]/2)
    discretized_pd = create_grid(pd, resolution, global_max)
    gabor_kernel = cv2.getGaborKernel((kernel_x, kernel_y), 
                                    sigma, theta, lambd, gamma, psi, 
                                    ktype=cv2.CV_32F)
    real_gabor = (cv2.filter2D(discretized_pd, -1, gabor_kernel)).flatten()
    return real_gabor, 0




class Signals:
    """
    A class for transforming persistence diagrams using various functional representations such as Fourier, Wavelet, and Gabor.
    
    Attributes:
        resolution (Tuple[int, int]): The resolution of the grid.
        global_max (Optional[Tuple[float, float]]): The maximum values for scaling the grid.
        energy (bool): Flag to indicate if energy should be calculated.
        wave (str): The wavelet type.
        sigma (float): Gabor filter sigma value.
        lambd (float): Gabor filter lambda value.
        gamma (float): Gabor filter gamma value.
        theta (float): Gabor filter theta value.
        psi (float): Gabor filter psi value.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the Signals class with default or specified parameters.

        Parameters:
            **kwargs: Variable keyword arguments for class properties.
        """
        self._defaults = {
            "function": "WAVELET",
            "resolution": (32, 32),
            "global_max": None,
            "energy": False,
            "wave": "coif1",
            "sigma": 5,
            "lambd": 10,
            "gamma": 0.5,
            "theta": 0,
            "psi": 0
        }
        self._update_properties(kwargs)
        self._set_embedding_function()
        self.fitted = False

    def _update_properties(self, properties: Dict[str, Any]) -> None:
        """
        Updates class properties based on provided arguments.

        Parameters:
            properties (Dict[str, Any]): Property values to update.
        """
        for key, value in properties.items():
            if key in self._defaults and value is not None:
                self._defaults[key] = value
        self.__dict__.update(self._defaults)

    def _set_embedding_function(self) -> None:
        """
        Sets the embedding function based on the specified method.
        """
        function_mappings = {
            "FOURIER": partial(fourier_functional, 
                               resolution=self.resolution,
                               global_max=self.global_max),
            "WAVELET": partial(wavelet_functional, 
                               resolution=self.resolution, 
                               global_max=self.global_max, 
                               wave=self.wave),
            "GABOR": partial(gabor_functional, 
                             resolution=self.resolution, 
                             global_max=self.global_max, 
                             sigma=self.sigma, 
                             lambd=self.lambd,
                             gamma=self.gamma, 
                             theta=self.theta, 
                             psi=self.psi)
        }
        self.embedding = function_mappings.get(self.function, None)
        if not self.embedding:
            raise ValueError(f"Invalid function specified: {self.function}")

    def __call__(self, pd: np.ndarray) -> np.ndarray:
        """
        Calls the embedding function on a persistence diagram.

        Parameters:
            pd (np.ndarray): A persistence diagram.

        Returns:
            np.ndarray: The result of applying the embedding function to the persistence diagram.
        """
        if not self.embedding:
            raise ValueError("No embedding function set. Please specify a valid function.")
        return self.embedding(pd)[0]

    def fit(self, pds: List[np.ndarray]) -> 'Signals':
        """
        Fits the model to a set of persistence diagrams.

        Parameters:
            pds (List[np.ndarray]): A list of persistence diagrams.

        Returns:
            Signals: The fitted Signals object.
        """
        if not self.embedding:
            raise ValueError("No embedding function set. Please specify a valid function.")

        concatenated_vectorization = []
        concatenated_energy = []
        for pd in pds:
            vectors, energy = self.embedding(pd)
            concatenated_energy.append(energy)
            concatenated_vectorization.extend(vectors)

        self.concatenated_vector = np.array(concatenated_vectorization)
        self.concatenated_energy = np.array(concatenated_energy)
        self.fitted = True
        return self

    def transform(self, pds: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transforms persistence diagrams after fitting.

        Parameters:
            pds (Optional[List[np.ndarray]]): A list of persistence diagrams to transform.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: The transformed data and, if requested, the energy.
        """
        if not self.fitted:
            raise RuntimeError("Transform called before fit. Please fit the model first.")

        return (self.concatenated_vector, self.concatenated_energy) if self.energy else (self.concatenated_vector)

