import numpy as np
from abc import ABC, abstractmethod

class BaseSaver(ABC):
    @abstractmethod
    def save_result(self, result_dict):
        pass

class FileSaver(BaseSaver):
    def __init__(self, save_path="results/experiment.csv"):
        self.save_path = save_path

    def save_result(self, result_dict):
        error = result_dict.get("preference_error", 999)
        np.savetxt(self.save_path, [[error]])
        print(f"Saved preference error = {error} to {self.save_path}")
