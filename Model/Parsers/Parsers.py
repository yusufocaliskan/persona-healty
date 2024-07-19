from pydantic import BaseModel, Field


class GuidanceChainParser(BaseModel):
    agent_index: int = Field(description="next agent to run")
    agent_name: str = Field(
        description="Name of chosen agent without space for example; 'ConversationAgent'"
    )
    reason: str = Field(description="reason to select this agent")
