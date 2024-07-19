from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser,OutputFixingParser
from langchain.chat_models import ChatOpenAI
from langchain.llms import BaseLLM
from Model.Config.Config import config
from Model.Parsers.Parsers import GuidanceChainParser


class BaseChain(LLMChain):
    """Child class of LLMChain. Have a class method to return a LLMChain object with
    given inputs
    """

    @classmethod
    def runChain(
        cls,
        llm: BaseLLM,
        chain_name: str,
        verbose: bool = False,
        use_parser: bool = False,
    ) -> LLMChain:
        """Class method which has been inherited from LLMChain class.

        Args:
            llm (BaseLLM): Langchain LLM Object to use
            chain_name (str): Name of the chain created
            verbose (bool, optional): Show inner dialog of chain. Defaults to True.

        Returns:
            LLMChain: Chain with given prompt template, llm type and verbose choice
        """

        # Read config and get information according to chain name
        chain_config = config["chains"][chain_name]
        messages = [
            SystemMessagePromptTemplate.from_template(
                chain_config["prompt_template"]["system_prompt_template"]
            ),
            HumanMessagePromptTemplate.from_template(
                chain_config["prompt_template"]["human_prompt_template"]
            ),
        ]
        prompt_input = chain_config["prompt_inputs"]

        if use_parser:
            # print(globals()[chain_name + "Parser"])
            parser = PydanticOutputParser(pydantic_object=globals()[chain_name + "Parser"])

            # Prompt template of chain
            prompt = ChatPromptTemplate(
                messages=messages,
                input_variables=prompt_input,
                output_parser=parser,
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )
        else:
            # Prompt template of chain
            prompt = ChatPromptTemplate(messages=messages, input_variables=prompt_input)

        # Returns a LLMChain object which can be used to get response from OpenAI ChatGPT.
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class Chain(object):
    """Chain class to be used in Teach AI Agents' chains. All chains in the application
    inherits this class.
    """

    def __init__(self, chain_name, **kwargs):
        """Constructor of Chain class.

        Args:
            chain_name (str): Name of the chain. Must be same with config file.
            **kwargs (dict): Keyword arguments. See below

            Keyword arguments:
                model_name: Initial model name of the chain. Defaults to config file
                temperature: Initial temperature value of the chain. Defaults to config file
                verbose: Initial verbose value of the chain. Defaults to config file
                use_parser: Initial parser requirement bool. Determine wheter to use parse or not. Defaults to config file.
        """
        self._chain_name = chain_name
        self._chain_config = config["chains"][chain_name]

        # Kwargs parameters with default values
        self._model_name = kwargs.get("model_name", self._chain_config["model_name"])
        self._temperature = kwargs.get("temperature", self._chain_config["temperature"])
        self._verbose = kwargs.get("verbose", self._chain_config["verbose"])
        self._use_parser = kwargs.get("use_parser", self._chain_config["use_parser"])

    @property
    def chain_name(self):
        return self._chain_name

    @property
    def chain_config(self):
        return self._chain_config

    @property
    def model_name(self):
        return self._model_name

    @property
    def temperature(self):
        return self._temperature

    @property
    def verbose(self):
        return self._verbose
    
    @property
    def require_parser(self):
        return self._use_parser

    def run(self, inputs, **kwargs) -> (BaseModel, LLMChain):
        """Runs the chain according to inputs and keyword arguments

        Args:
            inputs (dict): Required inputs for chain's prompt template.
            **kwargs (dict): Keyword arguments. See below

            Keyword arguments:
                model_name(str): Initial model name of the chain. Defaults to config file
                temperature(float): Initial temperature value of the chain. Defaults to config file
                verbose(bool): Initial verbose value of the chain. Defaults to config file

        Returns:
            (response, chain) (tuple): Response is string answer from chain. Chain is the used chain object
        """
        
        # Creating LLM
        llm = ChatOpenAI(
            model_name=kwargs.get("model_name", self._model_name),
            openai_api_key=config["OPEN_AI_API_KEY"],
            temperature=kwargs.get("temperature", self._temperature),
        )

        # Create chain from BaseChain
        chain = BaseChain.runChain(
            llm=llm,
            chain_name=self._chain_name,
            verbose=kwargs.get("verbose", self._verbose),
            use_parser=self._use_parser,
        )

        # Getting answer with inputs from created chain
        answer = None
        if self._use_parser:
            answer = chain.predict_and_parse(**inputs)
        else:
            answer = chain.run(**inputs)
        # Return both answer and chain itself
        return answer, chain
