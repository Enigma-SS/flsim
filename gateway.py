import client
import load_data
import logging
import numpy as np
import random
import os
import sys
from threading import Thread
import torch

class Gateway(object):
    """Gateway on the middle level."""

    def __init__(self, gateway_id):
        self.gateway_id = gateway_id
        self.clients = []

    def add_client(self, client_id):
        if client_id in self.clients:
            logging.info('Add Error: client {} already associated with gateway {}!'.format(
                client_id, self.gateway_id
            ))
        else:
            self.clients.append(client_id)

    def remove_client(self, client_id):
        if client_id in self.clients:
            self.clients.remove(client_id)
        else:
            logging.info('Remove Error: client {} is not associated with gateway {}!'.format(
                client_id, self.gateway_id
            ))

    def set_link_to_cloud(self, config):
        # Set the Gaussian distribution for link speed in Kbytes
        self.speed_min = config.network.cloud_gateway.get('min')
        self.speed_max = config.network.cloud_gateway.get('max')
        self.speed_mean = random.uniform(self.speed_min, self.speed_max)
        self.speed_std = config.network.cloud_gateway.get('std')

        # Set model size in Kbytes
        model_path = config.paths.model + '/global'
        if os.path.exists(model_path):
            self.model_size = os.path.getsize(model_path) / 1e3  # model size in Kbytes
        else:
            self.model_size = config.model.size  # estimated model size in Kbytes

    def set_delay_to_cloud(self):
        # Set the link speed and delay for the upcoming run
        link_speed = random.normalvariate(self.speed_mean, self.speed_std)
        link_speed = max(min(link_speed, self.speed_max), self.speed_min)
        self.delay = self.model_size / link_speed  # upload delay in sec

    def configure(self, config):
        import fl_model  # pylint: disable=import-error

        # Extract from config
        model_path = self.model_path = config.paths.model

        # Download most recent global model
        path = model_path + '/global'
        self.model = fl_model.Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

        # Set delay
        self.set_delay_to_cloud()

