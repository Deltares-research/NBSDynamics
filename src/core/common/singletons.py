"""
This file is intended to contain all the common classes used as unique object
throughout the `NBSDynamics` project.
Although these classes are defined elsewhere, here we implement them as singletons.
"""
from typing import Optional, Type

from src.core.common.constants import _Constants
from src.core.common.space_time import DataReshape


class RESHAPE(DataReshape):
    """
    `DataReshape` Singleton.
    """

    _instance: Optional[DataReshape] = None

    def __new__(cls: Type[DataReshape], **kwargs) -> DataReshape:
        """
        Overriding new method to ensure this class behaves as s singleton.

        Args:
            cls (Type[Constants]): Required instance class.

        Returns:
            Constants: Singleton for `CommonConstants`
        """
        if cls._instance == None:
            cls._instance = DataReshape(**kwargs).__new__(cls)
        return cls._instance


class CommonConstants(_Constants):
    """
    _Constants Singleton.
    """

    _instance: Optional[_Constants] = None

    def __new__(cls: Type[_Constants], **kwargs) -> _Constants:
        """
        Overriding new method to ensure this class behaves as s singleton.

        Args:
            cls (Type[_Constants]): Required instance class.

        Returns:
            _Constants: Singleton for `CommonConstants`
        """
        if cls._instance == None:
            cls._instance = _Constants(**kwargs).__new__(cls)
        return cls._instance
