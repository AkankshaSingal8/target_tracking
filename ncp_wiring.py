import os
from typing import Iterable, Dict
import tensorflow as tf
import kerasncp as kncp
from kerasncp.tf import LTCCell, WiredCfcCell
from tensorflow import keras


import numpy as np
from matplotlib.image import imread
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns


DEFAULT_NCP_SEED = 22222
wiring = kncp.wirings.NCP(
    inter_neurons=18,  # Number of inter neurons
    command_neurons=12,  # Number of command neurons
    motor_neurons=4,  # Number of motor neurons
    sensory_fanout=6,  # How many outgoing synapses has each sensory neuron
    inter_fanout=4,  # How many outgoing synapses has each inter neuron
    recurrent_command_synapses=4,  # Now many recurrent synapses are in the
    # command neuron layer
    motor_fanin=6,  # How many incoming syanpses has each motor neuron,
    seed=DEFAULT_NCP_SEED,  # random seed to generate connections between nodes
)

rnn_cell = LTCCell(wiring)

    
sns.set_style("white")
plt.figure(figsize=(6, 4))
legend_handles = rnn_cell.draw_graph(draw_labels=True)
plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()