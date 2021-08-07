import abc


class ClusterManager(metaclass=abc.ABCMeta):
    pass


class Cluster(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get(self):
        pass


class NodePool(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get(self):
        pass
