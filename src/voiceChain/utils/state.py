from enum import Enum, auto

class AgentState(Enum):
    """
    Defines the possible states of the conversation agent.
    """
    IDLE = auto()
    PROCESSING = auto()
    RESPONDING = auto()