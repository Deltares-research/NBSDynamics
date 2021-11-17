from pydantic import BaseModel as PydanticBaseModel
from pydantic import Extra


class BaseModel(PydanticBaseModel):
    """
    Custom definition of pydantic base model. This class helps 'NBSDynamics' model create and validation.
    """

    class Config:
        arbitrary_types_allowed = True


class ExtraModel(PydanticBaseModel):
    """
    Custom definition of pydantic base model. This class helps 'NBSDynamics' model create and validation.
    """

    class Config:
        """
        Allows this model to have extra fields defined during runtime.
        """

        arbitrary_types_allowed = True
        extra = Extra.allow
