import argparse
import config
import logging
import os
import server
from datetime import datetime
import time
import random
import numpy as np
import torch

# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./config.json',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')

args = parser.parse_args()

# Set logging
logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, args.log.upper()), datefmt='%H:%M:%S')


def main():
    """Run a federated learning simulation."""
    # Set random seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(10)

    # Read configuration file
    fl_config = config.Config(args.config)

    # Initialize server
    fl_server = {
        "basic": server.Server(fl_config),
        "accavg": server.AccAvgServer(fl_config),
        "directed": server.DirectedServer(fl_config),
        "kcenter": server.KCenterServer(fl_config),
        "kmeans": server.KMeansServer(fl_config),
        "magavg": server.MagAvgServer(fl_config),
        #"dqn": server.DQNServer(fl_config),
        "dqntrain": server.DQNTrainServer(fl_config),
        "sync": server.SyncServer(fl_config),
        "async": server.AsyncServer(fl_config),
    }[fl_config.server]
    fl_server.boot()

    # Run federated learning
    fl_server.run()

    # Save and plot accuracy-time curve
    if fl_config.server == "sync" or fl_config.server == "async":
        d_str = datetime.now().strftime("%m-%d-%H-%M-%S")
        fl_server.records.save_record('{}_{}.csv'.format(
            fl_config.server, d_str
        ))
        fl_server.records.plot_record('{}_{}.png'.format(
            fl_config.server, d_str
        ))

    # Delete global model
    #os.remove(fl_config.paths.model + '/global')


if __name__ == "__main__":
    st = time.time()
    main()
    elapsed = time.time() - st
    logging.info('The program takes {} s'.format(
        time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))
    ))
