from typing import List
from attrs import define, field
from Model.Config.Config import config

@define
class Persona:
    description: str = field(init=False, default=config["Persona"]["description"])
    attributes: List[str] = field(init=False, default=config["Persona"]["attributes"])

