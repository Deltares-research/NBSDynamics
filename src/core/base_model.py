from pydantic import BaseModel as PydanticBaseModel


class BaseModel(PydanticBaseModel):
    """
    Custom definition of pydantic base model. This class helps 'NBSDynamics' model create and validation.
    """

    class Config:
        arbitrary_types_allowed = True
