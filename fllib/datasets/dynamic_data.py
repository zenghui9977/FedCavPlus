import logging

from fllib.datasets.base import FederatedDataset

logger = logging.getLogger(__name__)

class DynamicFederatedDataset(FederatedDataset):
    def __init__(self, data_name, trainset, testset, simulated, simulated_root, distribution_type, clients_id, class_per_client=2, alpha=0.9, min_size=1, delta_data_num=50, dynamic_alpha = 0.1):
        super().__init__(data_name, trainset, testset, simulated, simulated_root, distribution_type, clients_id, class_per_client, alpha, min_size)
        self.delta_data_num = delta_data_num
        self.dynamic_alpha = dynamic_alpha


        