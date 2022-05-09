"""A small script for comparing theory to simulation."""
# Imports
import numpy as np
import matplotlib.pyplot as plt
import nest
import scipy.integrate as integrate
import scipy


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
    lif_neurons = nest.Create("iaf_psc_exp", sim_params["neurons"],
                              params=neuron_params)

    # Noise
    noise = nest.Create("noise_generator")
    noise.set({"std": Sigma, "dt": sim_params['dt_noise']})

    # Spike Detector
    spikedetector = nest.Create("spike_recorder")

    # Multimeter
    multimeter = nest.Create("multimeter")
    multimeter.set(record_from=["V_m"])

    # Connections
    nest.Connect(noise, lif_neurons)
    nest.Connect(lif_neurons, spikedetector)
    nest.Connect(multimeter, lif_neurons)

    # Running
    nest.Simulate(sim_params['simtime'])

    # Firing Rate
    fr_sim = spikedetector.n_events/sim_params['simtime']

    # Spike Data
    dSD = spikedetector.get("events")
    evs = dSD["senders"]
    ts = dSD["times"]
    # ts = ts.astype('int')

    # Variance
    var_sim = np.var(np.diff(ts))

    # Firing Rate
    return fr_sim, var_sim, ts, evs


def theorize(sim_params, neuron_params):
    """Calculate the theoretical firing rate and variance for a LIF neuron."""
    # %% Firing Rate based on Brunel
    def f(x): return np.exp(x**2) * (1 + scipy.special.erf(x))

    # Outer Bounds
    outer_top = (neuron_params["V_th"] - sim_params["mean_mem"]) / \
        sim_params['std_mem'] / np.sqrt(2)
    outer_bottom = (neuron_params["V_reset"] - sim_params['mean_mem']) / \
        sim_params['std_mem'] / np.sqrt(2)

    # Iti
    integral = integrate.quad(f, outer_bottom, outer_top)
    mu = neuron_params['t_ref'] + neuron_params['tau_m'] * np.sqrt(np.pi) * \
        integral[0]

    # %% Variance
    def f(y, x): return np.exp(x**2) * np.exp(y**2) * \
        (1 + scipy.special.erf(y))**2

    # Inner Bounds
    integral = integrate.dblquad(f, outer_bottom, outer_top,
                                 lambda x: -5, lambda x: x)
    #
    var = 2 * np.pi * integral[0] * neuron_params['tau_m']**2
    std = np.sqrt(var)

    # CV
    cv = std/mu

    return mu, std, cv


def compute_statistics(sim_params, exp_params, ts, evs):
    """Compute the mu, std and cv for spiking neurons."""
    # Initializing variables
    iti_mu_all = np.zeros((sim_params["neurons"], exp_params['nr_windows']))
    iti_std_all = np.zeros((sim_params["neurons"], exp_params['nr_windows']))

    for n in range(sim_params["neurons"]):

        # Separating spike times
        n_ts = np.array([ts[i] for i, e in enumerate(evs) if e == n+1])

        # Calculating iti's
        for en, w in enumerate(np.arange(0, sim_params["simtime"] -
                                         exp_params["window_size"],
                                         exp_params["step_size"])):

            # Cropping to window
            low = n_ts[n_ts > w]
            spikes = low[low < (w + exp_params["window_size"])]

            # Iti
            iti = np.diff(spikes)

            # Mean
            iti_mu = np.mean(iti)
            iti_std = np.std(iti)

            # Saving
            iti_mu_all[n, en] = iti_mu
            iti_std_all[n, en] = iti_std

    # Average average and mean
    mu = np.mean(iti_mu_all, axis=0)
    std = np.mean(iti_std_all, axis=0)

    # CV
    cv = std / mu

    return mu, std, cv


def experiment(exp_params, sim_params, neuron_params):
    """Run experiment based on parameters defined in exp_params dictionary."""
    # Initializing
    mu_sim_list = []
    std_sim_list = []
    cv_sim_list = []

    mu_theo_list = []
    std_theo_list = []
    cv_theo_list = []

    # Nr windows per simulation
    exp_params['nr_windows'] = int((sim_params["simtime"] -
                                   exp_params["window_size"]) /
                                   exp_params["step_size"])

    # Looping through stds
    for std in exp_params["stds"]:
        sim_params["std_mem"] = std

        # Simulate
        fr, var, ts, evs = simulate(sim_params, neuron_params)

        # Compute statistics
        mu_sim, std_sim, cv_sim = compute_statistics(sim_params, exp_params,
                                                     ts, evs)

        # Theory
        mu_theo, std_theo, cv_theo = theorize(sim_params, neuron_params)

        # Saving
        # sim
        mu_sim_list.append(mu_sim)
        std_sim_list.append(std_sim)
        cv_sim_list.append(cv_sim)

        # theo
        mu_theo_list.append(mu_theo)
        std_theo_list.append(std_theo)
        cv_theo_list.append(cv_theo)

        print(std_theo)

    # Simulation
    mu_sim = np.hstack(mu_sim_list)
    std_sim = np.hstack(std_sim_list)
    cv_sim = np.hstack(cv_sim_list)

    # Theory
    mu_theo = np.repeat(np.array(mu_theo_list), exp_params['nr_windows'])
    std_theo = np.repeat(np.array(std_theo_list), exp_params['nr_windows'])
    cv_theo = np.repeat(np.array(cv_theo_list), exp_params['nr_windows'])

    return mu_sim, std_sim, cv_sim, mu_theo, std_theo, cv_theo


# Experiment Parameters
exp_params = {'window_size': 800,
              'step_size': 100,
              'stds': [30, 15],
              'nr_windows': None}

# Simulation Parameters
sim_params = {'dt_noise': 0.01,
              'sim_res': 0.01,
              'mean_mem': 0.0,
              'std_mem': 3,
              'simtime': 10000,
              'seed': 12,
              'neurons': 50}

# Neuron Parameter
neuron_params = {'C_m': 1.0,
                 't_ref': 0.1,
                 'V_reset': 0.0,
                 'tau_m': 10.0,
                 'V_th': 15.0,
                 'E_L': 0.0}


mu_sim, std_sim, cv_sim, mu_theo, std_theo, cv_theo = experiment(exp_params,
                                                                 sim_params,
                                                                 neuron_params)

# Plotting
simtime = sim_params['simtime']/1000
simtime_total = simtime*len(exp_params['stds'])
time_windows = np.linspace(0, simtime_total, len(mu_sim))
alpha = 0.7

# Sigma Plot
fig, ax = plt.subplots()
for en, i in enumerate(exp_params['stds']):
    ax.hlines([exp_params['stds'][en]], simtime*en, simtime*(en+1),
              label=f'sigma {en}', color='darkslateblue')
ax.set(ylabel='Membrane voltage std', xlabel='time')
ax.set_ylim(0, 50)
ax.legend()

# ITI Plot
fig, ax = plt.subplots()
# Simulation
ax.plot(time_windows, mu_sim, label="simulation", c='red', alpha=alpha)
ax.fill_between(time_windows, mu_sim, mu_sim+std_sim,
                color='darkorange', alpha=0.2)
ax.fill_between(time_windows, mu_sim, mu_sim-std_sim,
                color='darkorange', alpha=0.2)
# Theory
ax.plot(time_windows, mu_theo, label="theory", c='k', alpha=alpha)
ax.fill_between(time_windows, mu_theo, mu_theo+std_theo, color='grey',
                alpha=0.2)
ax.fill_between(time_windows, mu_theo, mu_theo-std_theo, color='grey',
                alpha=0.2)
# Labels
ax.set(ylabel='ITI', xlabel='time')
ax.set_ylim(-50, 100)
ax.legend()

# CV Plot
fig, ax = plt.subplots()
ax.plot(time_windows, cv_sim, label="simulation", c='teal', alpha=alpha)
ax.plot(time_windows, cv_theo, label="theory", c='green', alpha=alpha)
ax.set(ylabel='CV', xlabel='time')
ax.set_ylim(0, 3)
ax.legend()

plt.show()