"""
This file is intended to contain all the common classes used as unique object
throughout the `NBSDynamics` project.
Although these classes are defined elsewhere, here we implement them as singletons.
"""
from typing import Type

from src.core.constants import Constants
from src.core.environment import Environment


class CommonConstants(Constants):
    """
    Constants Singleton.
    """

    _instance: Constants = None

    def __new__(cls: Type[Constants]) -> Constants:
        """
        Overriding new method to ensure this class behaves as s singleton.

        Args:
            cls (Type[Constants]): Required instance class.

        Returns:
            Constants: Singleton for `CommonConstants`
        """
        if cls._instance == None:
            cls._instance = Constants().__new__(cls)
        return cls._instance


class CommonEnvironment(Environment):
    """
    Environment Singleton.
    """

    _instance: Environment = None

    def __new__(cls: Type[Environment]) -> Environment:
        """
        Overriding new method to ensure this class behaves as s singleton.

        Args:
            cls (Type[Environment]): Required instance class.

        Returns:
            Environment: Singleton for `CommonEnvironment`
        """
        if cls._instance == None:
            cls._instance = Environment().__new__(cls)
        return cls._instance
