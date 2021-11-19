"""
This file is intended to contain all the common classes used as unique object
throughout the `NBSDynamics` project.
Although these classes are defined elsewhere, here we implement them as singletons.
"""
from src.core.common.space_time import DataReshape


class Singleton(object):
    """
    Singleton class representing the design pattern.
    This class can be used for concepts that are not meant to change state during a simulation
    such as DataReshape, represented by RESHAPE.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance


class RESHAPE(Singleton, DataReshape):
    """
    `DataReshape` Singleton.
    """

    pass
