from .contracts import OBSERVATION_SCHEMA_VERSION
from .actions import Action, PhaseAction, TotalAction
from .network import Network
from pu_normalizer import PUNormalizer

__all__ = [
    "EnvironmentConfig", "MicrogridEnv", "OBSERVATION_SCHEMA_VERSION", "PUNormalizer",
    "Action", "PhaseAction", "TotalAction", "Network",
]


def __getattr__(name):
    """Preserve package-level environment imports without an import cycle."""
    if name == "EnvironmentConfig":
        from microgrid_env import EnvironmentConfig
        return EnvironmentConfig
    if name == "MicrogridEnv":
        from microgrid_env import MicrogridEnv
        return MicrogridEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
