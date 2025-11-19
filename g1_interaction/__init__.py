# SPDX-License-Identifier: MIT

import os

# Get the root directory of the project
G1_INTERACTION_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
G1_INTERACTION_ENVS_DIR = os.path.join(G1_INTERACTION_ROOT_DIR, 'g1_interaction', 'envs')

__version__ = "1.0.0"

