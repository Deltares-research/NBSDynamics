from typing import Protocol


class HydrodynamicProtocol(Protocol):
    def initiate(self):
        """
        Initiates the working model.

        Raises:
            NotImplementedError: When the model does not implement its own definition.
        """
        raise NotImplementedError

    def update(self, coral, stormcat: int):
        """
        Updates the model with the given parameters.

        Args:
            coral (Coral): Coral model to be used.
            stormcat (int): Category of storm to apply.

        Raises:
            NotImplementedError: When the model does not implement its own definition.
        """
        raise NotImplementedError

    def finalise(self):
        """
        Finalizes the model.

        Raises:
            NotImplementedError: When the model does not implement its own definition.
        """
        raise NotImplementedError
