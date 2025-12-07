from abc import ABC, abstractmethod
from typing import Literal

class RoleClassifier(ABC):

    @abstractmethod
    def classify(self, question: str) -> Literal["user", "assistant"]:
        ...

class UserRoleClassifier(RoleClassifier):
    
    def classify(self, question: str) -> Literal["user", "assistant"]:
        return "user"