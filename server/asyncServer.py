import logging
import pickle
import random
import torch
from queue import PriorityQueue
import os
import sys
from server import Server, gateway
from .record import Record, Profile
from .asyncEvent import asyncGwEvent

class AsyncServer(Server):
    """Asynchronous federated learning server."""

    def boot(self):
        logging.info('Booting {} server...'.format(self.config.server))

        model_path = self.config.paths.model
        total_clients = self.config.clients.total
        total_gateways = self.config.network.gateway_total

        # Add fl_model to import path
        sys.path.append(model_path)

        # Set up simulated server
        self.load_data()
        self.load_model()
        self.make_gateways(total_gateways)
        self.make_clients(total_clients, self.gateways)
        self.set_link()

        # Initiate client profile of loss and delay
        self.profile = Profile(total_clients)
        self.profile.set_primary_label([client.pref for client in self.clients])

    def make_clients(self, num_clients, gateways):
        super().make_clients(num_clients)

        # Add gateway-client association
        for client in self.clients:
            # Randomly associate client with a gateway
            gateway_id = random.randint(0, len(gateways) - 1)
            gateways[gateway_id].add_client(client.client_id)
            client.set_gateway(gateway_id)

    def make_gateways(self, num_gws):
        gateways = []
        for gateway_id in range(num_gws):
            # Create new gateway
            new_gw = gateway.Gateway(gateway_id, self.config)
            gateways.append(new_gw)

        self.gateways = gateways

    def set_link(self):
        # Set the network link between cloud and gateway
        speed_cloud_gateway = []
        for gateway in self.gateways:
            gateway.set_link_to_cloud(self.config)
            speed_cloud_gateway.append(gateway.speed_mean)

        logging.info('Speed cloud-gw distribution: {} Kbps'.format([s for s in speed_cloud_gateway]))

        speed_gateway_client = []
        for client in self.clients:
            client.set_link_to_gateway(self.config)
            speed_gateway_client.append(client.speed_mean)

        logging.info('Speed distribution: {} Kbps'.format([s for s in speed_gateway_client]))

    def configuration(self, sample_clients, sample_gateways):
        super().configuration(sample_clients)

        for gateway_id in sample_gateways:
            self.gateways[gateway_id].configure(self.config, sample_clients)

    def load_model(self):
        import fl_model  # pylint: disable=import-error

        model_path = self.config.paths.model
        model_type = self.config.model

        logging.info('Model: {}'.format(model_type))

        # Set up global model
        self.model = fl_model.Net()
        self.async_save_model(self.model, model_path, 0.0)

        # Extract flattened weights (if applicable)
        if self.config.paths.reports:
            self.saved_reports = {}
            self.save_reports(0, [])  # Save initial model

    # Run asynchronous federated learning
    def run(self):
        rounds = self.config.fl.rounds
        target_accuracy = self.config.fl.target_accuracy
        reports_path = self.config.paths.reports

        # Init async parameters
        self.alpha = self.config.async_params.alpha
        self.staleness_func = self.config.async_params.staleness_func

        # Init self accuracy records
        self.records = Record()

        if target_accuracy:
            logging.info('Training: {} rounds or {}% accuracy\n'.format(
                rounds, 100 * target_accuracy))
        else:
            logging.info('Training: {} rounds\n'.format(rounds))

        # Perform rounds of federated learning
        T_old = 0.0
        for round in range(1, rounds + 1):
            logging.info('**** Round {}/{} ****'.format(round, rounds))

            # Perform async rounds of federated learning with certain
            # grouping strategy
            self.rm_old_models(self.config.paths.model, T_old)
            accuracy, T_new = self.async_round(round, T_old)

            # Update time
            T_old = T_new

            # Break loop when target accuracy is met
            if target_accuracy and (accuracy >= target_accuracy):
                logging.info('Target accuracy reached.')
                break

        if reports_path:
            with open(reports_path, 'wb') as f:
                pickle.dump(self.saved_reports, f)
            logging.info('Saved reports: {}'.format(reports_path))

    def async_round(self, round, T_old):
        """Run one async round for T_async"""
        import fl_model  # pylint: disable=import-error
        target_accuracy = self.config.fl.target_accuracy

        # Select clients and associated gateways to participate in the round
        # Note, sample_clients and sample_gateways are lists of objects
        sample_clients, sample_gateways = self.selection()
        self.total_samples = sum([len(client.data) for client in sample_clients])

        # Create a queue at the server to store gateway update events
        queue = PriorityQueue()

        # Create async gateway events
        events = [asyncGwEvent(sample_clients, gateway, T_old)
                  for gateway in sample_gateways]
        for event in events:
            queue.put(event)

        # This async round will end after the slowest group completes one round
        last_aggregate_time = max([e.est_aggregate_time for e in events])

        logging.info('Global async round lasts until {} secs'.format(last_aggregate_time))

        # Start the asynchronous updates
        while not queue.empty():
            gwEvent = queue.get()
            select_gateway, associate_clients = gwEvent.gateway, gwEvent.clients
            download_time = gwEvent.download_time

            self.async_gateway_configuration(select_gateway, associate_clients, download_time)
            report = select_gateway.async_round(associate_clients, download_time)
            T_cur = download_time + select_gateway.delay

            # Update profile and plot
            #self.update_profile(reports)
            # Plot every plot_interval
            #if math.floor(T_cur / self.config.plot_interval) > \
            #        math.floor(T_old / self.config.plot_interval):
            #    self.profile.plot(T_cur, self.config.paths.plot)

            # Perform weight aggregation
            logging.info('Aggregating updates from gateway {}'.format(select_gateway))
            staleness = select_gateway.delay
            updated_weights = self.federated_async_aggregation(report, staleness)

            # Load updated weights
            fl_model.load_weights(self.model, updated_weights)

            # Extract flattened weights (if applicable)
            if self.config.paths.reports:
                self.save_reports(round, report)

            # Save updated global model
            self.async_save_model(self.model, self.config.paths.model, T_cur)

            # Test global model accuracy
            if self.config.clients.do_test:  # Get average accuracy from client reports
                accuracy = self.accuracy_averaging(report)
            else:  # Test updated model on server
                testset = self.loader.get_testset()
                batch_size = self.config.fl.batch_size
                testloader = fl_model.get_testloader(testset, batch_size)
                accuracy = fl_model.test(self.model, testloader)

            logging.info('Average accuracy: {:.2f}%\n'.format(100 * accuracy))
            self.records.append_record(T_cur, accuracy, self.throughput)
            # Return when target accuracy is met
            if target_accuracy and \
                    (self.records.get_latest_acc() >= target_accuracy):
                logging.info('Target accuracy reached.')
                return self.records.get_latest_acc(), self.records.get_latest_t()

            # Insert the next aggregation of the group into queue
            # if time permitted
            if T_cur + gwEvent.est_delay < last_aggregate_time:
                gwEvent.download_time = T_cur
                gwEvent.est_aggregate_time = T_cur + gwEvent.est_delay
                queue.put(gwEvent)

        return self.records.get_latest_acc(), self.records.get_latest_t()

    def selection(self):
        # Select devices to participate in round
        clients_per_round = self.config.clients.per_round
        select_type = self.config.clients.selection

        if select_type == 'random':
            # Select clients randomly
            sample_clients = [client for client in random.sample(
                self.clients, clients_per_round)]

        elif select_type == 'short_latency_first':
            # Select the clients with short latencies and random loss
            sample_clients = sorted(self.clients, key=lambda c: c.est_delay)
            sample_clients = sample_clients[:clients_per_round]
            print(sample_clients)

        elif select_type == 'short_latency_high_loss_first':
            # Get the non-negative losses and delays
            losses = [c.loss for c in self.clients]
            losses_norm = [l/max(losses) for l in losses]
            delays = [c.est_delay for c in self.clients]
            delays_norm = [d/max(losses) for d in delays]

            # Sort the clients by jointly consider latency and loss
            gamma = 0.2
            sorted_idx = sorted(range(len(self.clients)),
                                key=lambda i: losses_norm[i] - gamma * delays_norm[i],
                                reverse=True)
            print([losses[i] for i in sorted_idx])
            print([delays[i] for i in sorted_idx])
            sample_clients = [self.clients[i] for i in sorted_idx]
            sample_clients = sample_clients[:clients_per_round]
            print(sample_clients)

        else:
            raise ValueError("client select type not implemented: {}".format(select_type))

        # Find out the associated gateways
        sample_gateways_id = list(set([client.gateway_id for client in sample_clients]))
        sample_gateways = [self.gateways[gateway_id] for gateway_id in sample_gateways_id]

        return sample_clients, sample_gateways

    def async_gateway_configuration(self, gateway, associate_clients, download_time):
        """Download global model to gateway and set up gateway round delay"""
        gateway.async_global_configure(self.config, associate_clients, download_time)


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

    def async_save_model(self, model, path, download_time):
        path += '/global_' + '{}'.format(download_time)
        torch.save(model.state_dict(), path)
        logging.info('Saved global model: {}'.format(path))

    def rm_old_models(self, path, cur_time):
        for filename in os.listdir(path):
            try:
                model_time = float(filename.split('_')[1])
                if model_time < cur_time:
                    os.remove(os.path.join(path, filename))
                    logging.info('Remove model {}'.format(filename))
            except Exception as e:
                logging.debug(e)
                continue

    def update_profile(self, reports):
        for report in reports:
            self.profile.update(report.client_id, report.loss, report.delay,
                                self.flatten_weights(report.weights))