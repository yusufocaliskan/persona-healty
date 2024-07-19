# Imports
from typing import List, Dict, Union, Tuple

from tenacity import retry, stop_after_attempt

from Model.Chain.Chain import Chain
from Model.Config.Config import config
from Model.Parsers.Parsers import GuidanceChainParser
from Model.Persona.Persona import Persona


class BaseAgent:
    """Parent agent class.
    Other agents inherited from this class
    """

    def __init__(self, chain_names: List[str]):
        """Base Agent Constructor

        Args:
            chain_names (List[str]): Name of the chains we would like to use in
        this agent
        """
        self._chains = {
            chain_name: Chain(chain_name=chain_name) for chain_name in chain_names
        }

        # Name of the AI using this agent. We use this information in prompt templates
        self._ai_name: Union[str, None] = None

        # Memory of the AI using this agent. We use this information in prompt templates
        self._memory: Union[str, None] = None

        # Name of the agent. Generally, we use this information to separate different agents
        self._agent_name = self.__class__.__name__

        # Lastly, run the chain to follow chain usage
        self._last_run_chain: Union[str, None] = None

        # Text variable in the Human section. It is the last user input
        self._text: Union[str, None] = None

        # Persona variable to import Persona attributes
        self.__persona = Persona()

    @property
    def agent_name(self):
        return self._agent_name

    @property
    def chains(self):
        return self._chains

    @property
    def ai_name(self):
        return self._ai_name

    @ai_name.setter
    def ai_name(self, value):
        self._ai_name = value

    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, value):
        self._memory = value

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    def __str__(self):
        return self._agent_name

    def run_chain(
            self, chain_name: str, chain_params: Dict[str, Union[str, float, bool]]
    ) -> str:
        """Runs a chain with given name

        Args:
            chain_name (str): Name of the chain we would like to run
            chain_params (Dict[str, Union[str, float, bool]]): Parameters for given chain

        Returns:
            str: Answer of the chosen chain
        """

        # Change last run chain
        self._last_run_chain = self._chains[chain_name]

        inputs = {}
        inputs.update()
        if config["chains"][chain_name]["use_persona"]:
            inputs = {"persona_description": self.__persona.description,
                      "persona_attributes": self.__persona.attributes}

        # Create inputs' dictionary to feed to LLM
        inputs.update({
            input_name: getattr(self, f"{input_name}")
            if "inputs" not in chain_params or bool(chain_params["inputs"])
            else chain_params["inputs"]
            for input_name in config["chains"][chain_name]["prompt_inputs"] if input_name not in inputs
        })

        # Return answer from LLM
        return self._last_run_chain.run(
            inputs=inputs,
            **(chain_params["parameters"] if "parameters" in chain_params else {}),  # type: ignore
        )[0]

    def run_chains(
            self,
            chains_params=None,
    ) -> Tuple[List[str], List[Union[str, GuidanceChainParser]], List[str]]:
        """Runs all chains in the agent and returns answers from them.

        Args:
            chains_params(Dict[str, Dict[str, Dict[str, Union[str, float, bool]]]], optional): Dictionary of all chains
            inputs and parameters. If not given, run all chains according to class variables. See below for more info

        Returns:
            (List[str], List[str], List[str]): List of all answers with in order, List of answers of chains, List of
            all chains

        Example input:
        {
            -chain name-: {
                "inputs": {-chain inputs-},
                "parameters": {-chain parameters-},
            },
            -chain name-: {
                "inputs": {-chain inputs-},
                "parameters": {-chain parameters-},
            },
        }
        """

        # Temp variables
        if chains_params is None:
            chains_params = {}
        answers = []

        # For loop to iterate over chains in the agent
        for chain_name, chain in self._chains.items():
            # Appending chain name to list
            answers.append(chain_name)

            # Create chain parameters input
            chain_params = (
                chains_params[chain_name] if chain_name in chains_params else {}
            )

            # Run chain
            answer = self.run_chain(chain_name=chain_name, chain_params=chain_params)

            # Append answer from a chain
            answers.append(answer)

        # Get only response strings
        responses = answers[1::2]

        # Get only chain names
        chains = answers[::2]

        # Return
        return answers, responses, chains


class GuidanceAgent(BaseAgent):
    """Guidance Agent for AI teacher.
    Responsible for selecting other agents in AI teacher
    """

    def __init__(self, chain_names=config["agents"]["GuidanceAgent"]["chains"]):
        super().__init__(chain_names=chain_names)

    @retry(stop=stop_after_attempt(3))
    def chooseNextAgent(self) -> (int, str):
        """Chooses next agent to run for AI teacher

        Raises:
            Exception: _description_

        Returns:
            (int, str): agent index, agent name
        """
        parameters = {"temperature": 0.5}

        try:
            _, responses, _ = self.run_chains(
                chains_params={"GuidanceChain": {"parameters": parameters}}
            )
            agent_index = responses[0].agent_index
            agent_name = responses[0].agent_name

            # Check if the agent name has blank or not (we do not want blank)
            if not agent_name.isspace():
                agent_name = config["AGENTS_NAMES"][agent_index]
        except Exception as e:
            print(e)
            raise ValueError

        return agent_index, agent_name


class ConversationAgent(BaseAgent):
    """Consists of the Conversation and Feedback chain.
    ConversationChain is responsible for general conversations with user.
    FeedbackChain is responsible for returning feedback to user based on conversation
    """

    def __init__(self, chain_names=config["agents"]["ConversationAgent"]["chains"]):
        super().__init__(chain_names=chain_names)
