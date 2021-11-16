from collections import namedtuple
import json


class Config(object):
    """Configuration module."""

    def __init__(self, config):
        self.paths = ""
        # Load config file
        with open(config, 'r') as config:
            self.config = json.load(config)
        # Extract configuration
        self.extract()

    def extract(self):
        config = self.config

        # -- Clients --
        fields = ['total', 'per_round', 'label_distribution',
                  'do_test', 'test_partition', 'selection']
        defaults = (0, 0, 'uniform', False, None, 'random')
        params = [config['clients'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.clients = namedtuple('clients', fields)(*params)

        assert self.clients.per_round <= self.clients.total

        # -- Data --
        fields = ['loading', 'partition', 'IID', 'bias', 'shard']
        defaults = ('static', 0, True, None, None)
        params = [config['data'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.data = namedtuple('data', fields)(*params)

        # Determine correct data loader
        assert self.data.IID ^ bool(self.data.bias) ^ bool(self.data.shard)
        if self.data.IID:
            self.loader = 'basic'
        elif self.data.bias:
            self.loader = 'bias'
        elif self.data.shard:
            self.loader = 'shard'

        # -- Federated learning --
        fields = ['rounds', 'target_accuracy', 'task', 'epochs', 'batch_size']
        defaults = (0, None, 'train', 0, 0)
        params = [config['federated_learning'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.fl = namedtuple('fl', fields)(*params)

        # -- Model --
        fields = ['name', 'size']
        defaults = ('MNIST', 1600)
        params = [config['model'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.model = namedtuple('model', fields)(*params)

        # -- Paths --
        fields = ['data', 'model', 'reports', 'plot']
        defaults = ('./data', './models', None, './plots')
        params = [config['paths'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        # Set specific model path
        params[fields.index('model')] += '/' + self.model.name

        self.paths = namedtuple('paths', fields)(*params)

        # -- Server --
        self.server = config['server']

        # -- Async --
        fields = ['alpha', 'staleness_func']
        defaults = (0.9, 'constant')
        params = [config['async_params'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.async_params = namedtuple('async_params', fields)(*params)

        # -- Network --
        fields = ['gateway_total', 'cloud_gateway', 'gateway_client']
        defaults = (5, 0, 0)
        params = [config['network'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.network = namedtuple('network', fields)(*params)

        # -- Plot interval --
        self.plot_interval = config['plot_interval']
