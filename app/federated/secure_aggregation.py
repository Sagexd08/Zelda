"""
Secure aggregation for privacy-preserving federated learning
"""
import numpy as np
from typing import Dict, List, Optional
from secrets import SystemRandom
import logging

logger = logging.getLogger(__name__)


class SecureAggregation:
    """
    Secure multiparty aggregation with random masking
    """
    
    def __init__(self, num_clients: int):
        """
        Args:
            num_clients: Number of participating clients
        """
        self.num_clients = num_clients
        self.random = SystemRandom()
    
    def generate_masks(self, shape: tuple, num_shares: int) -> List[np.ndarray]:
        """
        Generate secret shares using additive secret sharing
        
        Args:
            shape: Shape of the secret
            num_shares: Number of shares to generate
        
        Returns:
            List of shares
        """
        shares = []
        secret = np.random.randn(*shape)
        
        # Generate random shares
        for i in range(num_shares - 1):
            shares.append(np.random.randn(*shape))
        
        # Last share is the secret minus sum of others
        last_share = secret - sum(shares)
        shares.append(last_share)
        
        return shares
    
    def aggregate_secret_shares(self, shares: List[np.ndarray]) -> np.ndarray:
        """
        Recover secret from shares
        
        Args:
            shares: List of secret shares
        
        Returns:
            Recovered secret
        """
        return sum(shares)  # modulo arithmetic in real implementation
    
    def pairwise_mask_exchange(self, client_updates: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Apply pairwise masking for secure aggregation
        
        Args:
            client_updates: Dictionary of client_id -> update
        
        Returns:
            Masked aggregate
        """
        # Generate pairwise random keys
        pairwise_keys = {}
        client_ids = list(client_updates.keys())
        
        for i, client_i in enumerate(client_ids):
            for client_j in client_ids[i+1:]:
                key = np.random.randn(*list(client_updates.values())[0].shape)
                pairwise_keys[(client_i, client_j)] = key
        
        # Apply masking
        masked_updates = {}
        for client_id, update in client_updates.items():
            masked = update.copy()
            
            # Add/subtract pairwise keys
            for (ci, cj), key in pairwise_keys.items():
                if client_id == ci:
                    masked += key
                elif client_id == cj:
                    masked -= key
            
            masked_updates[client_id] = masked
        
        # Aggregate (masks cancel out)
        aggregate = sum(masked_updates.values())
        
        return aggregate


def create_secure_aggregation(num_clients: int) -> SecureAggregation:
    """
    Create secure aggregation instance
    
    Args:
        num_clients: Number of clients
    
    Returns:
        SecureAggregation instance
    """
    return SecureAggregation(num_clients)

