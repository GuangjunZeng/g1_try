# SPDX-License-Identifier: MIT

from g1_interaction import G1_INTERACTION_ROOT_DIR, G1_INTERACTION_ENVS_DIR
from g1_interaction.envs.base.retarget_robot import RetargetRobot
from g1_interaction.envs.g1.g1_retarget_config import G1RetargetCfg, G1RetargetCfgPPO

import os
from g1_interaction.utils.task_registry import task_registry

# Register the G1 retargeting task
task_registry.register("g1_retarget", RetargetRobot, G1RetargetCfg(), G1RetargetCfgPPO())

