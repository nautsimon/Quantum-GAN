from qiskit.aqua.components.uncertainty_models import UniformDistribution, UnivariateVariationalDistribution
from qiskit import BasicAer
from qiskit.aqua.components.initial_states import Custom
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.components.neural_networks import NumPyDiscriminator
from qiskit.aqua.components.neural_networks.quantum_generator import QuantumGenerator
from qiskit.aqua.algorithms import QGAN
from qiskit.circuit.library import TwoLocal
from qiskit.aqua.components.optimizers import ADAM
from qiskit import QuantumRegister, QuantumCircuit
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.nn as nn
from torch import optim
import random
import os
import argparse
import time
from IPython.display import HTML
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline


start = time.time()
#pylint: disable=no-member


# Root directory for dataset
dataroot = "./data/land"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 1

#   size using a transformer.
image_size = 64

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
#device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(len(dataset))
# Plot some training images
real_batch = next(iter(dataloader))[0].numpy()
print(real_batch)
reshaped_data = real_batch.reshape(-1)
bounds = np.array([-1., 1.])

print(reshaped_data)
print(type(reshaped_data))
#reshaped_norm_data = [t+1 for t in reshaped_data]
num_qubits = [4]
k = len(num_qubits)


# Set number of training epochs
# Note: The algorithm's runtime can be shortened by reducing the number of training epochs.
num_epochs = 10
# Batch size

# Initialize qGAN
qgan = QGAN(reshaped_data, bounds=bounds, num_qubits=num_qubits,
            batch_size=1, num_epochs=num_epochs, snapshot_dir="data")
print("QGAN set")
qgan.seed = 1
# Set quantum instance to run the quantum generator
quantum_instance = QuantumInstance(
    backend=BasicAer.get_backend('statevector_simulator'))
print("quantum_instance set")

# Set entangler map
entangler_map = [[0, 1]]


# Set an initial state for the generator circuit
init_dist = UniformDistribution(sum(num_qubits), low=bounds[0], high=bounds[1])
q = QuantumRegister(sum(num_qubits), name='q')
qc = QuantumCircuit(q)
init_dist.build(qc, q)
init_distribution = Custom(num_qubits=sum(num_qubits), circuit=qc)
var_form = TwoLocal(int(np.sum(num_qubits)), 'ry', 'cz', entanglement=entangler_map,
                    reps=1, initial_state=init_distribution)
# Set generator's initial parameters
init_params = aqua_globals.random.rand(
    var_form.num_parameters_settable) * 2 * np.pi
# Set generator circuit
g_circuit = UnivariateVariationalDistribution(int(sum(num_qubits)), var_form, init_params,
                                              low=bounds[0], high=bounds[1])

print("g_circuit set")

# Set quantum generator
qgan.set_generator(generator_circuit=g_circuit)
# Set classical discriminator neural network
discriminator = NumPyDiscriminator(len(num_qubits))
qgan.set_discriminator(discriminator)
