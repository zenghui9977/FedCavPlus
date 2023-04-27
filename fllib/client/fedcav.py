import logging
import copy
import time
import numpy as np
import torch
from fllib.client.base import BaseClient
import torchmetrics

logger = logging.getLogger(__name__)

CLIENT_ACC = 'train_acc'
CLIENT_LOSS = 'train_loss'

class FedCavClient(BaseClient):

    def __init__(self, config, device):
        super(FedCavClient, self).__init__(config, device)
        self.inference_loss = None

    def compute_inference_loss(self, last_global_model, local_trainset, loss_fn):
        last_global_model.eval()
        last_global_model.to(self.device)
        
        with torch.no_grad():
            batch_loss = []
            for imgs, labels in local_trainset:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                outputs = self.global_model(imgs)
                batch_loss.append(loss_fn(outputs, labels).item())
   
            total_loss = np.mean(batch_loss)

            logger.info('Inference Loss: {:.4f}'.format(total_loss))
        
            return total_loss

    def train(self, client_id, local_trainset):

        '''
        the local training process of FedCav
        '''
        start_time = time.time()
        loss_fn, optimizer = self.train_preparation()
        train_accuracy = torchmetrics.Accuracy().to(self.device)
        
        last_global_model = copy.deepcopy(self.local_model)

        self.inference_loss = self.compute_inference_loss(last_global_model, local_trainset, loss_fn)
        inference_time = time.time() - start_time

        for e in range(self.config.local_epoch):
            batch_loss = []
            train_accuracy.reset()
            for imgs, labels in local_trainset:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.local_model(imgs)

                # Loss and model parameters update
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

                _ = train_accuracy(outputs, labels)
                
            current_epoch_loss = np.mean(batch_loss)
            # current_epoch_acc = float(correct)/float(self.train_datasize)
            current_epoch_acc = train_accuracy.compute().item()

            self.train_records[CLIENT_LOSS].append(current_epoch_loss)
            self.train_records[CLIENT_ACC].append(current_epoch_acc)
            logger.debug('Client: {}, local epoch: {}, loss: {:.4f}, acc: {:.4f}'.format(client_id, e, current_epoch_loss, current_epoch_acc))

        train_time = time.time() - start_time
        logger.debug('Client: {}, inference time {:.4f}s, training {:.4f}s'.format(client_id, inference_time, train_time))


    def upload(self):
        return (self.local_model, self.inference_loss)
    
    