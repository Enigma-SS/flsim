import load_data
import logging
import numpy as np
import random
import os
from queue import PriorityQueue
import torch
from .asyncEvent import asyncClEvent

class Gateway(object):
    """Gateway on the middle level."""

    def __init__(self, gateway_id, config):
        self.gateway_id = gateway_id
        self.clients_id = []
        self.config = config

    def add_client(self, client_id):
        if client_id in self.clients_id:
            logging.info('Add Error: client {} already associated with gateway {}!'.format(
                client_id, self.gateway_id
            ))
        else:
            self.clients_id.append(client_id)

    def remove_client(self, client_id):
        if client_id in self.clients_id:
            self.clients_id.remove(client_id)
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

        # Set estimated delay
        self.est_delay_gw_to_cloud = self.model_size / self.speed_mean

    def set_delay_to_cloud(self, sample_clients):
        # Set the link speed and delay for the upcoming run
        link_speed = random.normalvariate(self.speed_mean, self.speed_std)
        link_speed = max(min(link_speed, self.speed_max), self.speed_min)
        self.delay = self.model_size / link_speed  # upload delay in sec

        # Add maximum client delay
        max_client_delay = max([client.est_delay for client in sample_clients])
        self.delay += max_client_delay

    def async_global_configure(self, config, sample_clients, download_time):
        import fl_model  # pylint: disable=import-error

        # Init async parameters
        self.alpha = self.config.async_params.alpha
        self.staleness_func = self.config.async_params.staleness_func

        # Extract from config
        model_path = self.model_path = config.paths.model

        # Download most recent global model
        path = model_path + '/global_{}'.format(download_time)
        self.model = fl_model.Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        logging.info('Load global model: {}'.format(path))

        # Save the model to be the latest gateway model
        self.async_save_gateway_model(self.model, model_path, download_time)

        # Set delay
        self.set_delay_to_cloud(sample_clients)

    def async_local_configure(self, select_client, download_time):
        #loader_type = self.config.loader
        #loading = self.config.data.loading

        #if loading == 'dynamic':
            # Create shards if applicable
        #    if loader_type == 'shard':
        #        self.loader.create_shards()

        # Configure selected clients for federated learning task
        #for client in sample_clients:
            #if loading == 'dynamic':
            #    self.set_client_data(client)  # Send data partition to client

        # Extract config for client
        config = self.config

        # Continue configuration on client
        select_client.async_local_configure(config, download_time)

    def async_round(self, sample_clients, download_time):
        """Run one async round until the slowest client finish one round"""
        import fl_model

        self.total_samples = sum([len(client.data) for client in sample_clients])

        # Create a queue at the server to store gateway update events
        queue = PriorityQueue()

        # Create async client events
        events = [asyncClEvent(client, download_time) for client in sample_clients]
        for event in events:
            queue.put(event)

        # This async round will end after the slowest group completes one round
        last_aggregate_time = max([e.est_aggregate_time for e in events])
        logging.info('Gateway {} async round lasts until {} secs'.format(
            self.gateway_id, last_aggregate_time))

        while not queue.empty():
            event = queue.get()
            select_client = event.client

            # async client configuration
            self.async_local_configure(select_client, event.download_time)

            # Run on the client
            select_client.run(reg=True)
            T_cur = event.download_time + select_client.delay
            # Request report on weights, loss, delay, throughput
            report = select_client.get_report()

            # Perform weight aggregation
            logging.info('Aggregating updates on gateway {} from clients {}'.format(
                self.gateway_id, select_client.client_id))
            staleness = select_client.delay
            updated_weights = self.federated_async_aggregation(report, staleness)

            # Load updated weights and save as the latest model
            fl_model.load_weights(self.model, updated_weights)
            self.async_save_gateway_model(self.model, self.config.paths.model, T_cur)

            # Insert the next aggregation of the group into queue if time permitted
            if T_cur + select_client.est_delay < last_aggregate_time:
                event.download_time = T_cur
                event.est_aggregate_time = T_cur + select_client.est_delay
                queue.put(event)

        gateway_weights = fl_model.extract_weights(self.model)
        gateway_report = Report(self, gateway_weights, sample_clients, T_cur)

        return gateway_report

    def federated_async_aggregation(self, report, staleness):
        import fl_model  # pylint: disable=import-error

        # Extract updates from the report
        weights = report.weights

        # Perform weighted averaging
        new_weights = [torch.zeros(x.size())  # pylint: disable=no-member
                      for _, x in weights[0]]
        num_samples = report.num_samples
        for j, (_, weight) in enumerate(weights):
            # Use weighted average by number of samples
            new_weights[j] += weight * (num_samples / self.total_samples)

        # Extract baseline model weights - latest model
        baseline_weights = fl_model.extract_weights(self.model)

        # Calculate the staleness-aware weights
        alpha_t = self.alpha * self.staleness(staleness)
        logging.info('{} staleness: {} alpha_t: {}'.format(
            self.staleness_func, staleness, alpha_t
        ))

        # Load updated weights into model
        updated_weights = []
        for i, (name, weight) in enumerate(baseline_weights):
            updated_weights.append(
                (name, (1 - alpha_t) * weight + alpha_t * new_weights[i])
            )

        return updated_weights

    def staleness(self, staleness):
        if self.staleness_func == "constant":
            return 1
        elif self.staleness_func == "polynomial":
            a = 0.5
            return pow(staleness+1, -a)
        elif self.staleness_func == "hinge":
            a, b = 10, 4
            if staleness <= b:
                return 1
            else:
                return 1 / (a * (staleness - b) + 1)

    def async_save_gateway_model(self, model, path, download_time):
        path += '/gateway{}_{}'.format(self.gateway_id, download_time)
        torch.save(model.state_dict(), path)
        logging.info('Saved gateway {} model: {}'.format(self.gateway_id, path))


class Report(object):
    """Federated learning client report."""

    def __init__(self, gateway, weights, sample_clients, finish_time):
        self.gateway_id = gateway.client_id
        self.weights = weights
        self.num_samples = sum([len(client.data) for client in sample_clients])
        self.finish_time = finish_time

