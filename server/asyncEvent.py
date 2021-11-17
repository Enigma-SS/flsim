
class asyncGwEvent(object):
    """Basic async event."""
    def __init__(self, clients, gateway, download_time):
        # Pick the clients that are associated with the given gateway object
        self.clients = [client for client in clients
                        if client.client_id in gateway.clients_id]
        self.gateway = gateway
        self.download_time = download_time
        max_client_delay = max([client.est_delay for client in self.clients])
        self.est_delay = max_client_delay + self.gateway.est_delay_gw_to_cloud
        self.est_aggregate_time = self.download_time + self.est_delay

    def __eq__(self, other):
        return self.est_aggregate_time == other.est_aggregate_time

    def __ne__(self, other):
        return self.est_aggregate_time != other.est_aggregate_time

    def __lt__(self, other):
        return self.est_aggregate_time < other.est_aggregate_time

    def __le__(self, other):
        return self.est_aggregate_time <= other.est_aggregate_time

    def __gt__(self, other):
        return self.est_aggregate_time > other.est_aggregate_time

    def __ge__(self, other):
        return self.est_aggregate_time >= other.est_aggregate_time


class asyncClEvent(object):
    """Basic async event."""
    def __init__(self, client, download_time):
        self.client = client
        self.download_time = download_time
        self.est_aggregate_time = self.download_time + self.client.est_delay

    def __eq__(self, other):
        return self.est_aggregate_time == other.est_aggregate_time

    def __ne__(self, other):
        return self.est_aggregate_time != other.est_aggregate_time

    def __lt__(self, other):
        return self.est_aggregate_time < other.est_aggregate_time

    def __le__(self, other):
        return self.est_aggregate_time <= other.est_aggregate_time

    def __gt__(self, other):
        return self.est_aggregate_time > other.est_aggregate_time

    def __ge__(self, other):
        return self.est_aggregate_time >= other.est_aggregate_time