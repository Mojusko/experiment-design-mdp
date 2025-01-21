import numpy as np
from abc import ABC, abstractmethod
from typing import List

class BaseSaver(ABC):
    @abstractmethod
    def save_result(self, result_dict):
        pass

class FileSaver(BaseSaver):
    def __init__(self, params=None):
        self.params = params or {}

    def save_result(self, result_dict):
        error = result_dict.get("preference_error", 999)
        np.savetxt(self.params.path, [[error]])
        print(f"Saved preference error = {error} to {self.params.path}")

    def save_visits(self, visits: List[List[str]]):
        with open(self.params.log, "w") as f:
            for visit in visits:
                f.write(" ".join(visit) + "\n")
        print(f"Saved visits to {self.params.log}")