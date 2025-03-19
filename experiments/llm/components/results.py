import torch

class ExperimentResults:
    """Container for all experiment results with standardized access methods."""
    
    def __init__(self):
        self.metrics = {}        # Test metrics (preference_error, cosine_error, etc.)
        self.visits = None       # All exploration visits
        self.estimator = None    # Trained estimator model
        self.metadata = {}       # Any other experiment metadata
        
    def add_metric(self, name, value):
        """Add a metric value to results"""
        self.metrics[name] = value
        
    def add_metrics(self, metrics_dict):
        """Add multiple metrics from a dictionary"""
        self.metrics.update(metrics_dict)
        
    def set_visits(self, visits):
        """Set the exploration visits"""
        self.visits = visits
        
    def set_estimator(self, estimator):
        """Set the trained estimator"""
        self.estimator = estimator
        
    def add_metadata(self, key, value):
        """Add experiment metadata"""
        self.metadata[key] = value
        
    def get_theta(self):
        """Extract theta from estimator if available"""
        if self.estimator and hasattr(self.estimator, 'theta_ml'):
            return self.estimator.theta_ml()
        return None
