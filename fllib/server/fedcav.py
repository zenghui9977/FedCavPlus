import os
import logging

from fllib.server.base import BaseServer
from fllib.server.aggeration import FedCav_Aggregation

logger = logging.getLogger(__name__)

class FedCavServer(BaseServer):
    def __init__(self, config, clients, client_class, global_model, fl_trainset, testset, device, current_round=0, records_save_filename=None, vis=None):
        super().__init__(config, clients, client_class, global_model, fl_trainset, testset, device, current_round, records_save_filename, vis)
        self.clients_inference_loss = []

    def client_training(self):
        if len(self.selected_clients) > 0:
            for client in self.selected_clients:
                (local_update, inference_loss) = self.client_class.step(global_model=self.global_model,
                                                      client_id=client, 
                                                      local_trainset=self.fl_trainset.get_dataloader(client, batch_size=self.train_batchsize),
                                                      )
                self.local_updates[client] = {
                    'model': local_update.state_dict(),
                    'size': self.fl_trainset.get_client_datasize(client_id=client),
                    'inference_loss': inference_loss
                }                                          
        else:
            logger.warning('No clients in this round')
            self.local_updates = None

        return self.local_updates
    
    def aggregation(self):
        if self.local_updates is None:
            self.aggregated_model_dict = self.global_model.state_dict()
        else:
            self.aggregated_model_dict = FedCav_Aggregation(self.local_updates)
        return self.aggregated_model_dict
