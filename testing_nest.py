#!/usr/bin/env python3
"""Small script to test how fast nest can simulate."""

import numpy as np
import nest
import matplotlib.pyplot as plt
import time


def simulate(sim_params, neuron_params):
    """Simulate neurons according to sim and neuron parameters."""
    # Simulation
    nest.set_verbosity("M_WARNING")
    nest.ResetKernel()
    nest.rng_seed = sim_params['seed']
    nest.resolution = sim_params['sim_res']

    neurons = nest.Create("iaf_psc_alpha", sim_params["neurons"],
                          params=neuron_params)

    # Noise
    noise = nest.Create("noise_generator")
    noise.set({"std": 20, "dt": sim_params['dt_noise']})

    # Spike Detector
    spikedetector = nest.Create("spike_recorder")

    # Connections
    nest.Connect(noise, neurons)
    nest.Connect(neurons, spikedetector)

    # Running
    nest.Simulate(sim_params['simtime'])

    # Spike Data
    dSD = spikedetector.get("events")
    evs = dSD["senders"]
    ts = dSD["times"]
    fr = spikedetector.n_events/sim_params['simtime']*1000

    return ts, evs, fr


# Simulation Parameters
sim_params = {'dt_noise': 0.01,
              'sim_res': 0.01,
              'mean_mem': -65.0,
              'std_mem': 20,
              'simtime': 10000,
              'seed': 18,
              'neurons': 200,
              'search_start': [-73, 20],
              'neuron_model': 'lif',
              'to_optimize': 'sim'}

# Neuron Parameter
neuron_params = {'C_m': 1.0,
                 't_ref': 0.1,
                 'V_reset': -65.0,
                 'tau_m': 10.0,
                 'V_th': -50.0,
                 'E_L': -65.0}

time_vec = []
nr_neurons = np.linspace(1, 200, 20)

ts, evs, fr = simulate(sim_params, neuron_params)
print(fr)

# for n in nr_neurons:
#     sim_params['neurons'] = int(n)
#     start = time.time()
#     ts, evs, fr = simulate(sim_params, neuron_params)
#     end = time.time()
#     time_vec.append(end-start)
#     print('ey')

# plt.plot(nr_neurons,time_vec)
# plt.show()
