from typing import List, Dict
from time import time


class MemoryVariable:
    """Custom memory variable for teacher AI.
    """
    def __init__(
        self, by: str, message: str, agent_name: str, timestamp: float = time()
    ):
        """Constructor of MemoryVariable class

        Args:
            by (str): String representation of sender of message. For example; User or teacher name.
            message (str): Message string
            agent_name (str): Agent who answered with message. If message belongs to user, none or empty can be used.
            timestamp (float, optional): Message send time. Defaults to creation time.
        """
        self._by = by
        self._message = message
        self._agent_name = agent_name
        self._timestamp = timestamp

    @property
    def by(self):
        return self._by

    @property
    def message(self):
        return self._message

    @property
    def agent_name(self):
        return self._agent_name

    @property
    def timestamp(self):
        return self._timestamp


class BaseMemory:
    """Base class to be used in different AI teacher applications. It can store and manipulate
    memory variables.
    """
    def __init__(self):
        self._memory_array: List[MemoryVariable] = []
        self._number_of_memories = 0

    def __str__(self):
        text = ""
        if self.number_of_memories > 0:
            text = "----------------------------------------\n".join(
                [f"{memory.by}: {memory.message}\n" for memory in self.memory_array]
            ) + "----------------------------------------\n"
            
        return text

    @property
    def memory_array(self):
        return self._memory_array

    def append(self, by: str, message: str, **kwargs):
        self._memory_array.append(
            MemoryVariable(
                by=by,
                message=message,
                agent_name=kwargs.get("agent_name", ""),
                timestamp=kwargs.get("timestamp", time()),
            )
        )

    @memory_array.deleter
    def memory_array(self):
        self._memory_array.pop()

    @property
    def number_of_memories(self):
        return len(self._memory_array)

    
    def get_last_user_memory(self, user_name: str = "User"):
        """Returns last memory variable send by User

        Args:
            user_name (str, optional): Name of the User in case it is not named as "User". Defaults to "User".

        Returns:
            MemoryVariable: MemoryVariable class
        """
        last_user_memory = None
        for i in range(1, self.number_of_memories + 1):
            if self._memory_array[-1 * i].by == user_name:
                last_user_memory = self._memory_array[-1 * i]

        return last_user_memory
    
    def get_all_agents(self):
        """Returns all agents in messaging order

        Returns:
            List[str]: List of agents name with order
        """
        return [memory._agent_name for memory in self.memory_array  if memory.agent_name not in ["", "User"]]
    
    def get_last_agent_memory(self, agent_name: str):
        """Returns last agent memory variable

        Args:
            agent_name (str): Name of the agent

        Returns:
            MemoryVariable: MemoryVariable class
        """
        last_agent_memory = None
        for i in range(1, self.number_of_memories + 1):
            if self._memory_array[-1 * i].agent_name == agent_name:
                last_agent_memory = self._memory_array[-1 * i].message

        return last_agent_memory