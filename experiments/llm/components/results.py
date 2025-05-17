import torch

class ExperimentResults:
    """Container for all experiment results with standardized access methods."""
    
    def __init__(self):
        self.metrics = {}         # Test metrics (preference_error, cosine_error, etc.)
        self.visits = None        # All exploration visits
        self.estimators = None    # List of trained estimator models (one per scorer model)
        self.feedbacks = None     # List of feedback objects
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

    def set_feedbacks(self, feedbacks: list):
        """Set the list of feedback objects"""
        self.feedbacks = feedbacks

    def get_estimators(self) -> list:
        """Get the list of trained estimators"""
        return self.estimators

    def add_metadata(self, key, value):
        """Add experiment metadata"""
        self.metadata[key] = value

    def get_thetas(self) -> list:
        """Extract learned theta from each feedback object in the list if available"""
        thetas = []
        if self.feedbacks:
            for feedback in self.feedbacks:
                if feedback:
                    thetas.append(feedback.get_learned_theta())
                else:
                    thetas.append(None) # Append None if feedback object is missing
        else:
            # Fallback or warning if feedbacks list itself is not set
            # This case should ideally be handled by ensuring set_feedbacks is called.
            # For robustness, one might iterate self.estimators if self.feedbacks is None,
            # but the new abstraction is via feedback.
            pass # Or log a warning: print("Warning: Feedbacks list not set in ExperimentResults for get_thetas.")
        return thetas
