from typing import Callable, Optional, Tuple

import numpy as np


class CoralOnly:
    """Execute functions only in the presence of corals."""

    spacetime = None

    @property
    def space(self):
        """Space dimension."""
        return None if self.spacetime is None else self.spacetime[0]

    @property
    def time(self):
        """Time dimension."""
        return None if self.spacetime is None else self.spacetime[1]

    def in_space(
        self,
        coral,
        function: Callable,
        args: Tuple,
        no_cover_value: Optional[float] = 0,
    ) -> np.ndarray:
        """
        Only execute the function when there is coral cover.

        Args:
            coral (Coral): Coral object.
            function (Callable): Function to be executed.
            args (Tuple): input arguments of the function.
            no_cover_value (Optional[float], optional): Default value in absence of coral cover. Defaults to 0.

        Raises:
            ValueError: When sizes do not match.

        Returns:
            np.ndarray: Result of the coral function.
        """
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, (float, int)) or (
                isinstance(arg, np.ndarray) and not arg.shape
            ):
                args[i] = np.repeat(arg, self.space)
            elif not len(arg) == self.space:
                msg = f"Sizes do not match up, {len(arg)} =/= {self.space}."
                raise ValueError(msg)

        output = no_cover_value * np.ones(self.space)
        output[coral.cover > 0] = function(*[arg[coral.cover > 0] for arg in args])
        return output

    def in_spacetime(
        self,
        coral,
        function: Callable,
        args: Tuple,
        no_cover_value: Optional[float] = 0,
    ):
        """Only execute the function when there is coral cover.

        :param coral: coral object
        :param function: function to be executed
        :param args: input arguments of the function
        :param no_cover_value: default value in absence of coral cover

        :type coral: Coral
        :type args: tuple
        :type no_cover_value: float, optional
        """
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, (float, int)) or (
                isinstance(arg, np.ndarray) and not arg.shape
            ):
                args[i] = arg * np.ones(self.spacetime)
            elif arg.shape == coral.cover.shape:
                args[i] = np.tile(arg, (self.time, 1)).transpose()
            elif not arg.shape == self.spacetime:
                msg = f"Sizes do not match up, {arg.shape} =/= {self.spacetime}."
                raise ValueError(msg)

        output = no_cover_value * np.ones(self.spacetime)
        output[coral.cover > 0] = function(*[arg[coral.cover > 0] for arg in args])
        return output
