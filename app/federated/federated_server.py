"""
Federated learning server for secure aggregation
"""
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class FederatedServer:
    """
    Central server for federated learning aggregation
    """
    
    def __init__(self, num_clients: int, aggregation_method: str = "fedavg"):
        """
        Args:
            num_clients: Number of participating clients
            aggregation_method: Aggregation method (fedavg, fedprox, etc.)
        """
        self.num_clients = num_clients
        self.aggregation_method = aggregation_method
        self.global_model = None
        self.client_updates = defaultdict(dict)
        self.round_number = 0
    
    def initialize_global_model(self, model_weights: Dict):
        """Initialize global model weights"""
        self.global_model = model_weights
        logger.info("Global model initialized")
    
    def receive_client_update(self, client_id: str, weights: Dict, num_samples: int):
        """
        Receive model update from client
        
        Args:
            client_id: Unique client identifier
            weights: Client model weights
            num_samples: Number of training samples
        """
        self.client_updates[self.round_number][client_id] = {
            'weights': weights,
            'num_samples': num_samples
        }
        logger.info(f"Received update from client {client_id}")
    
    def aggregate(self) -> Optional[Dict]:
        """
        Aggregate client updates to global model
        
        Returns:
            Aggregated global model weights
        """
        if not self.client_updates[self.round_number]:
            logger.warning("No client updates to aggregate")
            return None
        
        if self.aggregation_method == "fedavg":
            return self._fedavg_aggregation()
        elif self.aggregation_method == "fedprox":
            return self._fedprox_aggregation()
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def _fedavg_aggregation(self) -> Dict:
        """
        Federated Averaging aggregation
        
        Returns:
            Averaged weights
        """
        # Compute total samples
        total_samples = sum(
            update['num_samples']
            for update in self.client_updates[self.round_number].values()
        )
        
        # Weighted average
        aggregated_weights = {}
        for client_id, update in self.client_updates[self.round_number].items():
            weight = update['num_samples'] / total_samples
            
            for key, value in update['weights'].items():
                if key not in aggregated_weights:
                    aggregated_weights[key] = np.zeros_like(value)
                aggregated_weights[key] += weight * value
        
        # Update global model
        self.global_model = aggregated_weights
        logger.info("FedAvg aggregation completed")
        
        return aggregated_weights
    
    def _fedprox_aggregation(self) -> Dict:
        """
        FedProx aggregation with proximal term
        
        Returns:
            Proximized averaged weights
        """
        # Similar to FedAvg but with proximal term
        # TODO: Implement FedProx specific logic
        return self._fedavg_aggregation()
    
    def get_global_model(self) -> Optional[Dict]:
        """Get current global model"""
        return self.global_model
    
    def increment_round(self):
        """Move to next training round"""
        self.round_number += 1
        logger.info(f"Starting round {self.round_number}")


def create_federated_server(num_clients: int, method: str = "fedavg") -> FederatedServer:
    """
    Create federated learning server
    
    Args:
        num_clients: Number of clients
        method: Aggregation method
    
    Returns:
        FederatedServer instance
    """
    return FederatedServer(num_clients, method)

