import random
from dataclasses import dataclass
from typing import List, Optional, Dict

from Model.Agents.Agents import GuidanceAgent, ConversationAgent, BaseAgent
from Model.Config.Config import config
from Model.Memory.Memory import BaseMemory


@dataclass
class AgentIndex:
    min: int
    max: int


class PersonaChatbotMemory(BaseMemory):
    """AI PersonaChatbot memory class. Inherits from BaseMemory"""

    def __init__(self):
        super().__init__()


class BasePersonaChatbot:
    """Base AI PersonaChatbot class.
    Can
        * store memory,
        * different agents with different chains,
        * run an agent
    """

    def __init__(self, personachatbot_name: str, agents: List[BaseAgent]):
        """Constructor for Base PersonaChatbot Class

        Args:
            personachatbot_name (str): Name of the AI teacher
            agents (List[BaseAgent]): Agents to be used in the teacher
        """
        # Assign teacher name

        self._ai_name: str = personachatbot_name

        # Create memory
        self._memory: BaseMemory = PersonaChatbotMemory()

        # Last user input
        self._user_input: Optional[str] = None

        self._agents: Dict[str, BaseAgent] = {}

        # Assign agents
        for agent in agents:
            agent.ai_name = self._ai_name
            agent.memory = self._memory
            agent.text = self._user_input
            self._agents[agent.agent_name] = agent

        self._number_of_agents = len([*self._agents])
        # User name
        self._user_name = "User"

    @property
    def user_name(self):
        return self._user_name

    @property
    def memory(self):
        return self._memory

    @user_name.setter
    def user_name(self, value):
        self._user_name = value

    @property
    def user_input(self):
        return self._user_input

    @user_input.setter
    def user_input(self, value: str):
        """User input set method. Changes user input class variable.
        Also add memory a new message by user

        Args:
            value (str): _description_
        """
        self._user_input = value
        self._memory.append(by=self._user_name, message=value)

    @property
    def number_of_agents(self):
        self._number_of_agents = len([*self._agents])
        return self._number_of_agents

    @property
    def agents(self):
        return self._agents

    def __str__(self):
        text = f"{self._ai_name}\n" + "\n".join(
            [
                f"{index + 1}. {agent_name}"
                for index, agent_name in enumerate([*self._agents])
            ]
        )
        return text

    def start(self):
        """Returns a random starting conversation string from config file. Also
        adds chosen conversation to memory.

        Returns:
            str: random starting conversation for AI teacher
        """
        # Choose a random starting text from config file and change AI name to
        # decide one
        # start_text = random.choice(
        #     config["PossibleTeacherStartingConversations"]
        # ).format(ai_name=self._ai_name)

        start_text = "Hey there. My name is Tim. How can I help you today?"

        # Add to memory
        self.add_to_memory(by=self._ai_name, message=start_text)

        return start_text

    def choose_next_agent(self) -> (int, str):
        """Determines next agent to answer user prompt according to conversation history.
        Uses GuidanceAgent to select next agent.

        Returns:
            (int, str): agent_index in AGENTS array, agent name
        """

        # Calls for GuidanceAgent which is responsible from selecting next agent to
        # answer user prompt, according to conversation history.
        guidance_agent: GuidanceAgent = self._agents["GuidanceAgent"]
        agent_index, agent_name = guidance_agent.chooseNextAgent()

        return agent_index, agent_name

    def run_agent_by_name(self, agent_name: str) -> str:
        """Run agent by name. The Name of the agent must be same with config file naming

        Args:
            agent_name (str): Name of agent

        Returns:
            str: Answer of agent
        """
        # Change text variable in the agent
        self._agents[agent_name].text = self._user_input

        _, responses, _ = self._agents[agent_name].run_chains()

        message = "\n".join(responses)
        self.add_to_memory(by=self._ai_name, message=message, agent_name=agent_name)
        return message

    def add_to_memory(self, by: str, message: str, **kwargs):
        """Appends to the memory array.

        Args:
            by (str): Message owner. Can be user or teacher
            message (str): Message string

            Keyword arguments:
                agent_name (str): Name of agent used to produce a message. Defaults to ""
                timestamp (float): Timestamp value. Defaults to current time.
        """

        # Append a message to memory
        self._memory.append(
            by=by,
            message=message,
            agent_name=kwargs.get("agent_name", None),
            timestamp=kwargs.get("timestamp", None),
        )

    def main_loop(self):
        print(f"{self._ai_name}: {self.start()}\n------------------------------------------------")
        while True:
            self.user_input = input("User Input: ")
            print("----------------------------------------------------------------")
            _, agent_name = self.choose_next_agent()
            answer = self.run_agent_by_name(agent_name=agent_name)
            print(
                f"{self._ai_name}: {answer}\n------------------------------------------------"
            )


class PersonaChatbot(BasePersonaChatbot):
    """Language teacher of application. Inherits from BasePersonaChatbot class"""

    def __init__(self, personachatbot_name: str, **kwargs):
        """Constructor for PersonaChatbot class

        Args:
            personachatbot_name (str): Name of the PersonaChatbot
            **kwargs (dict): Keyword arguments. See below

            Keyword arguments:
                * Agents (List(BaseAgent)): Agents to be used in PersonaChatbot. Order is important.
            Defaults to [GuidanceAgent(), ConversationAgent(), ExamAgent(), GrammarAgent()]
        """

        # Agent
        agents = [GuidanceAgent(), ConversationAgent()]
        super().__init__(personachatbot_name=personachatbot_name, agents=kwargs.get("agents", agents))
