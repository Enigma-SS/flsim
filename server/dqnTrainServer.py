import logging
import torch
import numpy as np
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threading import Thread
from itertools import count
import matplotlib.pyplot as plt
import torch.optim as optim
from server import Server
from .record import Record, Profile
from .dqn import ReplayMemory, Transition, DQN, select_action, optimize_model

# Cuda settings
use_cuda = torch.cuda.is_available()
device = torch.device(  # pylint: disable=no-member
    'cuda' if use_cuda else 'cpu')

GAMMA = 0.999
TARGET_UPDATE = 10
WEIGHT_PCA_DIM = 100

class DQNTrainServer(Server):
    """Server for DQN training federated learning server."""

    def boot(self):
        logging.info('Booting {} server...'.format(self.config.server))

        model_path = self.config.paths.model
        total_clients = self.config.clients.total

        # Add fl_model to import path
        sys.path.append(model_path)

        self.n_actions = total_clients
        self.policy_net = DQN(n_input=WEIGHT_PCA_DIM*(total_clients+1),
                              n_output=total_clients).to(device=device, dtype=torch.float32)
        self.target_net = DQN(n_input=WEIGHT_PCA_DIM*(total_clients+1),
                              n_output=total_clients).to(device=device, dtype=torch.float32)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

        self.memory = ReplayMemory(10000)

    def make_clients(self, num_clients):
        super().make_clients(num_clients)

        # Set link speed for clients
        speed = []
        for client in self.clients:
            client.set_link(self.config)
            speed.append(client.speed_mean)

        logging.info('Speed distribution: {} Kbps'.format([s for s in speed]))

        # Initiate client profile of loss and delay
        self.profile = Profile(num_clients, self.loader.labels)
        if not self.config.data.IID:
            self.profile.set_primary_label([client.pref for client in self.clients])

    # Run DQN training
    def run(self):
        num_episodes = 200
        reward_episode = []
        for i_episode in range(num_episodes):
            logging.info('Starting episode {}...'.format(i_episode))
            # Init environment for episode
            state = self.episode_init()
            total_reward = 0.0

            # Start the episode
            for t in count():
                # Select and perform an action
                action = select_action(state, self.policy_net, self.n_actions,
                                       self.config.clients.per_round)
                next_state, reward, done = self.step(action)
                logging.info('eps {} step {} action {} reward {}'.format(i_episode, t,
                                                                         action.detach().cpu().numpy(),
                                                                         reward.item()))
                total_reward += np.power(GAMMA, t) * reward

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                optimize_model(self.policy_net, self.target_net, self.memory,
                               self.optimizer)
                if done:
                    reward_episode.append(total_reward)
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        # Save model and plot the training procedure
        torch.save(self.target_net.state_dict(), self.config.paths.model + './dqn')
        plt.figure(figsize=(6, 8))
        plt.plot(reward_episode)
        plt.xlabel('Episodes')
        plt.ylabel('Total Return')

    def episode_init(self):
        import fl_model

        # Set up simulated server
        total_clients = self.config.clients.total
        self.load_data()
        self.load_model()
        self.make_clients(total_clients)

        # Init the server and state for the current episode
        _ = [client.set_delay() for client in self.clients]
        self.configuration(self.clients)
        threads = [Thread(target=client.run) for client in self.clients]
        [t.start() for t in threads]
        [t.join() for t in threads]
        reports = self.reporting(self.clients)
        # Update profile and plot
        self.update_profile(reports)
        self.profile.plot(0.0, self.config.paths.plot)

        # Perform weight aggregation
        updated_weights = self.aggregation(reports)

        # Load updated weights to global model and save
        fl_model.load_weights(self.model, updated_weights)
        self.save_model(self.model, self.config.paths.model)

        # Train PCA and gather initial state (weights after PCA)
        self.weights_array = [self.flatten_weights(updated_weights)]
        for report in reports:
            self.weights_array.append(self.flatten_weights(report.weights))
        self.weights_array = np.array(self.weights_array)

        # Scaling the weights collected from all clients
        self.scaler = StandardScaler()
        self.scaler.fit(self.weights_array)
        self.weights_array_scale = self.scaler.transform(self.weights_array)

        # PCA transform
        self.pca = PCA(n_components=WEIGHT_PCA_DIM)
        self.pca.fit(self.weights_array_scale)

        state = torch.tensor(self.pca.transform(self.weights_array_scale),
                             device=device, dtype=torch.float32).view(1, -1)
        return state

    def step(self, action):
        """
        Perform one step of transform from state to next state, when action
        Args:
            - action: torch tensor of the selected action
        Return:
            - next_state: torch tensor of next state
            - reward: torch tensor of current reward
            - done: True if the episode is done (i.e., reach target accuracy)
        """
        import fl_model  # pylint: disable=import-error

        # Convert state and action to numpy
        action = action.view(-1).detach().cpu().numpy()

        # Configure sample clients
        sample_clients = [self.clients[i] for i in action]
        self.configuration(sample_clients)
        # Use the max delay in all sample clients as the delay in sync round
        _ = [client.set_delay() for client in sample_clients]
        max_delay = max([c.delay for c in sample_clients])

        # Run clients using multithreading for better parallelism
        threads = [Thread(target=client.run) for client in sample_clients]
        [t.start() for t in threads]
        [t.join() for t in threads]
        #T_cur = T_old + max_delay  # Update current time

        # Receive client updates
        reports = self.reporting(sample_clients)

        # Update profile and plot
        self.update_profile(reports)
        # Plot every plot_interval
        #if math.floor(T_cur / self.config.plot_interval) > \
        #        math.floor(T_old / self.config.plot_interval):
        #    self.profile.plot(T_cur, self.config.paths.plot)

        # Perform weight aggregation
        #logging.info('Aggregating updates')
        updated_weights = self.aggregation(reports)

        # Load updated weights
        fl_model.load_weights(self.model, updated_weights)

        # Save updated global model
        self.save_model(self.model, self.config.paths.model)

        # Update state
        self.weights_array[0] = self.flatten_weights(updated_weights)  # global weights
        for report in reports:
            self.weights_array[report.client_id] = self.flatten_weights(report.weights)
        self.weights_array_scale = self.scaler.transform(self.weights_array)
        next_state = torch.tensor(self.pca.transform(self.weights_array_scale),
                                  device=device, dtype=torch.float32).view(1, -1)

        # Test global model accuracy
        if self.config.clients.do_test:  # Get average accuracy from client reports
            accuracy = self.accuracy_averaging(reports)
        else:  # Test updated model on server
            testset = self.loader.get_testset()
            batch_size = self.config.fl.batch_size
            testloader = fl_model.get_testloader(testset, batch_size)
            accuracy = fl_model.test(self.model, testloader)

        logging.info('Average accuracy: {:.2f}%'.format(100 * accuracy))
        target_accuracy = self.config.fl.target_accuracy
        reward = torch.tensor([np.power(64, accuracy - target_accuracy) - 1]).to(device)

        # Finish one episode if target accuracy is met
        done = False
        if target_accuracy and (accuracy >= target_accuracy):
            logging.info('Target accuracy reached.')
            done = True

        return next_state, reward, done

    def update_profile(self, reports):
        for report in reports:
            self.profile.update(report.client_id, report.loss, report.delay,
                                self.flatten_weights(report.weights),
                                self.flatten_weights(report.grads))
