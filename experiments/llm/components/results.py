import torch

class ExperimentResults:
    """Container for all experiment results with standardized access methods."""
    
    def __init__(self):
        self.metrics = {}         # Test metrics (preference_error, cosine_error, etc.)
        self.visits = None        # All exploration visits
        self.estimators = None    # List of trained estimator models (one per scorer model)
        self.metadata = {}        # Any other experiment metadata

    def add_metric(self, name, value):
        """Add a metric value to results"""
        self.metrics[name] = value
        
    def add_metrics(self, metrics_dict):
        """Add multiple metrics from a dictionary"""
        self.metrics.update(metrics_dict)
        
    def set_visits(self, visits):
        """Set the exploration visits"""
        self.visits = visits

    def set_estimators(self, estimators: list):
        """Set the list of trained estimators"""
        self.estimators = estimators

    def get_estimators(self) -> list:
        """Get the list of trained estimators"""
        return self.estimators

    def add_metadata(self, key, value):
        """Add experiment metadata"""
        self.metadata[key] = value

    def get_thetas(self) -> list:
        """Extract theta from each estimator in the list if available"""
        thetas = []
        if self.estimators:
            for estimator in self.estimators:
                if estimator and hasattr(estimator, 'theta_ml'):
                    thetas.append(estimator.theta_ml())
                else:
                    thetas.append(None) # Append None if estimator or theta_ml is missing
        return thetas
