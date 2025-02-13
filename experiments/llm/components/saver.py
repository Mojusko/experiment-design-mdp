import numpy as np
import json
from abc import ABC, abstractmethod

class BaseSaver(ABC):
    @abstractmethod
    def save_result(self, result_dict):
        pass

class FileSaver(BaseSaver):
    def __init__(self, params=None):
        self.params = params or {}

    def save_result(self, result_dict):
        # Dump the entire result_dict to the specified file in JSON format
        with open(self.params.path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        print(f"Saved result with to {self.params.path}:")
        print(json.dumps(result_dict, indent=2))
        
