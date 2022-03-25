"""A small script for comparing theory to simulation."""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import nest
import scipy.integrate as integrate
import scipy
from tqdm import tqdm


# %% Theory vs Simulation
def simulate(sim_params, neuron_params):
    """Simulate the firing rate of a neuron using the nest simulator."""
    # dt noise correction
    Sigma = np.sqrt(2/(sim_params['dt_noise']*neuron_params['tau_m'])) * \
        neuron_params['C_m'] * sim_params['std_mem']

    # Simulation
    nest.set_verbosity("M_WARNING")
    nest.ResetKernel()
    nest.rng_seed = sim_params['seed']
    nest.resolution = sim_params['sim_res']

    # Neuron
    lif_neuron = nest.Create("iaf_psc_exp")
    lif_neuron.set(neuron_params)

    # Noise
    noise = nest.Create("noise_generator")
    noise.set({"std": Sigma, "dt": sim_params['dt_noise']})

    # Spike Detector
    spikedetector = nest.Create("spike_recorder")

    # Connections
    nest.Connect(noise, lif_neuron)
    nest.Connect(lif_neuron, spikedetector)

    # Running
    nest.Simulate(sim_params['simtime'])

    # Firing Rate
    fr_sim = spikedetector.n_events/sim_params['simtime']*1000

    # Spike Data
    dSD = nest.GetStatus(spikedetector, keys="events")[0]
    ts = dSD["times"]
    ts = ts.astype('int')

    # Variance
    var_sim = np.var(np.diff(ts))

    # Firing Rate
    return fr_sim, var_sim, ts


def theorize(sim_params, neuron_params):
    """Calculate the theoretical firing rate and variance for a LIF neuron."""
    # %% Firing Rate based on Brunel
    def f(x): return np.exp(x**2) * (1 + scipy.special.erf(x))

    # Outer Bounds
    outer_top = (neuron_params["V_th"] - sim_params["mean_mem"]) / \
        sim_params['std_mem']
    outer_bottom = (neuron_params["V_reset"] - sim_params['mean_mem']) / \
        sim_params['std_mem']

    # Iti
    integral = integrate.quad(f, outer_bottom, outer_top)
    iti = neuron_params['t_ref'] + neuron_params['tau_m'] * np.sqrt(np.pi) * \
        integral[0]

    # Firing Rate
    fr = 1/iti*1000

    # %% Variance
    def f(y, x): return np.exp(x**2) * np.exp(y**2) * \
        (1 + scipy.special.erf(y))**2

    # Inner Bounds
    def inner_top(x): return x
    def inner_bottom(x): return -5
    integral = integrate.dblquad(f, outer_bottom, outer_top,
                                 lambda x: -5, lambda x: x)
    var = fr ** 2 + 2 * np.pi * integral[0]

    return fr, var


def run(sim_params, neuron_params):
    """Run the simulation for a range of membrane voltage stds."""
    # Membrane stds
    stds = np.linspace(*sim_params['stds'])

    # Lists
    fr_theo_rec = []
    fr_sim_rec = []
    var_theo_rec = []
    var_sim_rec = []

    for std in tqdm(stds):

        # Setting membrane std
        sim_params['std_mem'] = std

        # Theory
        theo_fr, theo_var = theorize(sim_params, neuron_params)

        # Simulation
        sim_fr, sim_var, ts = simulate(sim_params, neuron_params)

        # Recording
        fr_theo_rec.append(theo_fr)
        fr_sim_rec.append(sim_fr)
        var_theo_rec.append(theo_var)
        var_sim_rec.append(sim_var)

    return fr_theo_rec, fr_sim_rec, var_theo_rec, var_sim_rec, stds


if __name__ == "__main__":

    # Simulation Parameters
    sim_params = {'dt_noise': 0.1,
                  'sim_res': 0.1,
                  'mean_mem': -70,
                  'std_mem': 10.0,
                  'simtime': 100000,
                  'stds': [6, 25, 100],
                  'seed': 7}

    # Neuron Parameter
    neuron_params = {"C_m": 1.0,
                     "t_ref": 2.0,
                     "V_reset": -70.0,
                     "tau_m": 20.0,
                     "V_th": -55.0}

    # Running Simulation
    fr_theo, fr_sim, var_theo, var_sim, stds = run(sim_params, neuron_params)

    # Plotting
    alpha = 0.7
    fig, ax = plt.subplots()
    ax.plot(stds, fr_theo, label='theory', color='k', alpha=alpha)
    ax.plot(stds, fr_sim, label='simulation', color='r', alpha=alpha)
    ax.set(ylabel='Firing Rate', xlabel='Voltage STD')
    ax.legend()

    alpha = 0.7
    fig, ax = plt.subplots()
    ax.plot(stds, var_theo, label='theory', color='k', alpha=alpha)
    ax.plot(stds, var_sim, label='simulation', color='red', alpha=alpha)
    ax.set(ylabel='Variance', xlabel='Voltage STD')
    ax.legend()

    plt.show()

    # # Look at single ts
    # fr, var, ts = simulate(sim_params, neuron_params)
    # iti_var = np.diff(ts)
    # plt.hist(iti_var,bins=100)
    # plt.show()
