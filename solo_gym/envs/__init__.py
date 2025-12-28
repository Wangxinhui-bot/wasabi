##
# Locomotion environments.
##
# fmt: off
from .base.legged_robot import LeggedRobot
from .solo8.solo8 import Solo8
from .solo8.solo8_config import (
    Solo8FlatCfg, 
    Solo8FlatCfgPPO
)
from .pi_plus.pi_plus import PiPlus
from .pi_plus.pi_plus_config import (
    PiPlusFlatCfg,
    PiPlusFlatCfgPPO
)

# fmt: on

##
# Task registration
##
from solo_gym.utils.task_registry import task_registry

task_registry.register("solo8", Solo8, Solo8FlatCfg, Solo8FlatCfgPPO)
task_registry.register("pi_plus", PiPlus, PiPlusFlatCfg, PiPlusFlatCfgPPO)