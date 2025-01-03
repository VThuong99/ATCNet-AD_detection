from abc import ABC, abstractmethod
import numpy as np
import mne

from src.rbps import relative_band_power

class Preprocessor(ABC):
    """
    Interface for preprocessing data.
    """
    @abstractmethod
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Processes the input data.

        Args:
            data: The input data to be processed (NumPy ndarray).

        Returns:
            The processed data (NumPy ndarray).
        """
        pass

    @property
    def name(self):
        """
        Returns the name of the preprocessor.
        """
        return self.__class__.__name__

class RawPreprocessor(Preprocessor):
    """
    Returns the original input data without any processing.
    """
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        return data

class PsdPreprocessor(Preprocessor):
    """
    Processes data using the PSD method.
    Requires the output of RawPreprocessor.
    """
    def __init__(self):
        self.raw_preprocessor = RawPreprocessor()
        self.sfreq = 500
        self.fmin = 0.5
        self.fmax = 45
        self.freqs = None

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        # Step 1: Preprocess raw data
        raw_data = self.raw_preprocessor.preprocess(data)
        print(f"Running {self.name}")

        # Step 2: Compute PSD for each epoch
        n_epochs, n_channels, n_times = raw_data.shape
        all_psds = []
        all_freqs: np.ndarray | None = None

        for epoch in range(n_epochs):
            psds, freqs = mne.time_frequency.psd_array_welch(
                raw_data[epoch],  # Shape (n_channels, n_times)
                sfreq=self.sfreq,
                fmin=self.fmin,
                fmax=self.fmax,
                verbose=False
            )
            all_psds.append(psds)
            if all_freqs is None:
                all_freqs = freqs  # Save frequencies from the first iteration

        # Convert list of PSDs into a single ndarray
        processed_data = np.array(all_psds)  # Shape: (n_epochs, n_channels, n_freqs)
        self.freqs = all_freqs
        return processed_data


class RbpPreprocessor(Preprocessor):
    """
    Processes data using the RBP method.
    Requires the output of PsdPreprocessor.
    """
    def __init__(self):
        self.psd_preprocessor = PsdPreprocessor()
        self.freq_bands = [0.5,4.0,8.0,13.0,25.0,45.0]

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        # First, ensure PSD preprocessing is done (which includes Raw)
        psd_data = self.psd_preprocessor.preprocess(data)

        rbps = relative_band_power(psd_data,self.psd_preprocessor.freqs,self.freq_bands) 

        rbps = np.array(rbps)
        return rbps
    
class FlattenPreprocessor(Preprocessor):
    """
    Flattens the input data.
    """
    def __init__(self, last_preprocessor: Preprocessor | None = None):
        if last_preprocessor is not None:
            self.last_preprocessor = last_preprocessor

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        if hasattr(self, "last_preprocessor"):
            data = self.last_preprocessor.preprocess(data)
        return data.reshape((data.shape[0], -1))