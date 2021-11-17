import logging
import sys
import pickle
import random
import math
from threading import Thread
from server import Server, gateway
from .record import Record, Profile

class SyncServer(Server):
    """Synchronous federated learning server."""

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

    def make_clients(self, num_clients, gateways):
        super().make_clients(num_clients)

        # Set link speed for clients
        speed = []
        for client in self.clients:
            # Randomly associate client with a gateway
            gateway_id = random.randint(0, len(gateways) - 1)
            gateways[gateway_id].add_client(client.client_id)
            client.set_gateway(gateway_id)

        logging.info('Speed distribution: {} Kbps'.format([s for s in speed]))

        # Initiate client profile of loss and delay
        self.profile = Profile(num_clients)
        self.profile.set_primary_label([client.pref for client in self.clients])

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
            self.gateways[gateway_id].configure(self.config)

    # Run synchronous federated learning
    def run(self):
        rounds = self.config.fl.rounds
        target_accuracy = self.config.fl.target_accuracy
        reports_path = self.config.paths.reports

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

            # Run the sync federated learning round
            accuracy, T_new = self.sync_round(round, T_old)
            logging.info('Round finished at time {} s\n'.format(T_new))

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

    def sync_round(self, round, T_old):
        import fl_model  # pylint: disable=import-error

        # Select clients to participate in the round
        sample_clients, sample_gateways = self.selection()

        # Configure sample clients, including delay and throughput
        self.configuration(sample_clients, sample_gateways)
        self.throughput = sum([client.throughput for client in sample_clients])
        logging.info('Avg throughput {} kB/s'.format(self.throughput))

        # Use the max delay in all sample clients as the delay in sync round
        max_delay = max([c.delay for c in sample_clients])

        # Run clients using multithreading for better parallelism
        threads = [Thread(target=client.run) for client in sample_clients]
        [t.start() for t in threads]
        [t.join() for t in threads]
        T_cur = T_old + max_delay  # Update current time

        # Receive client updates
        reports = self.reporting(sample_clients)

        # Update profile and plot
        self.update_profile(reports)
        # Plot every plot_interval
        if math.floor(T_cur / self.config.plot_interval) > \
                math.floor(T_old / self.config.plot_interval):
            self.profile.plot(T_cur, self.config.paths.plot)

        # Perform weight aggregation
        logging.info('Aggregating updates')
        updated_weights = self.aggregation(reports)

        # Load updated weights
        fl_model.load_weights(self.model, updated_weights)

        # Extract flattened weights (if applicable)
        if self.config.paths.reports:
            self.save_reports(round, reports)

        # Save updated global model
        self.save_model(self.model, self.config.paths.model)

        # Test global model accuracy
        if self.config.clients.do_test:  # Get average accuracy from client reports
            accuracy = self.accuracy_averaging(reports)
        else:  # Test updated model on server
            testset = self.loader.get_testset()
            batch_size = self.config.fl.batch_size
            testloader = fl_model.get_testloader(testset, batch_size)
            accuracy = fl_model.test(self.model, testloader)

        logging.info('Average accuracy: {:.2f}%'.format(100 * accuracy))
        self.records.append_record(T_cur, accuracy, self.throughput)
        return self.records.get_latest_acc(), self.records.get_latest_t()

    def selection(self):
        # Select devices to participate in round
        clients_per_round = self.config.clients.per_round

        # Select clients randomly
        sample_clients = [client for client in random.sample(
            self.clients, clients_per_round)]

        # Find out the associated gateways
        sample_gateways = list(set([client.gateway for client in sample_clients]))

        return sample_clients, sample_gateways

    def update_profile(self, reports):
        for report in reports:
            self.profile.update(report.client_id, report.loss, report.delay,
                                self.flatten_weights(report.weights))