"""
This file is intended to contain all the common classes used as unique object
throughout the `NBSDynamics` project.
Although these classes are defined elsewhere, here we implement them as singletons.
"""
from src.core.common.space_time import DataReshape


class Singleton(object):
    _instance = None

    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance


class RESHAPE(Singleton, DataReshape):
    """
    `DataReshape` Singleton.
    """

    pass
