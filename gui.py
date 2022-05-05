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
    lif_neurons = nest.Create("iaf_psc_exp", sim_params["neurons"], params=neuron_params)

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

    # print(ts)
    # print(evs)
    # print(dSD)

    # Firing Rate
    return fr_sim, var_sim, ts, evs

def compute_statistics(sim_params, exp_params, ts, evs):

    # Initializing variables
    neuron_ts = []
    nr_windows = int((sim_params["simtime"] - exp_params["window_size"]) / exp_params["step_size"])
    iti_mu_all = np.zeros((sim_params["neurons"],nr_windows))
    iti_std_all = np.zeros((sim_params["neurons"],nr_windows))

    for n in range(sim_params["neurons"]):

        # Separating spike times
        n_ts = np.array([ts[i] for i, e in enumerate(evs) if e == n+1])

        # Calculating iti's
        for en,w in enumerate(np.arange(0, sim_params["simtime"] - exp_params["window_size"] , exp_params["step_size"])):

            # Cropping to window
            low = n_ts[n_ts>w]
            spikes = low[low < (w + exp_params["window_size"])]

            # Iti
            iti = np.diff(spikes)

            # Mean
            iti_mu = np.mean(iti)
            iti_std = np.std(iti)


            # Saving
            iti_mu_all[n,en] = iti_mu
            iti_std_all[n,en] = iti_std

    # Average average and mean
    mu  = np.mean(iti_mu_all, axis=0)
    std = np.mean(iti_std_all, axis=0)

    # CV
    cv = std / mu

    return mu, std, cv

def experiment(exp_params,sim_params,neuron_params):

    # Initializing
    mu_exp_list = []
    std_exp_list = []
    cv_exp_list = []

    # Looping through stds
    for std in exp_params["stds"]:
        sim_params["std_mem"] = std

        # Simulate
        fr, var, ts, evs = simulate(sim_params, neuron_params)

        # Compute statistics
        mu, std, cv = compute_statistics(sim_params, exp_params, ts, evs)

        # Saving
        mu_exp_list.append(mu)
        std_exp_list.append(std)
        cv_exp_list.append(cv)

    mu_exp = np.hstack(mu_exp_list)
    std_exp = np.hstack(std_exp_list)
    cv_exp = np.hstack(cv_exp_list)


    return mu_exp, std_exp, cv_exp

# Experiment Parameters
exp_params = {'window_size': 800,
              'step_size': 100,
              'stds': [2,4,8]}

# Simulation Parameters
sim_params = {'dt_noise': 0.01,
                'sim_res': 0.01,
                'mean_mem': 0.0,
                'std_mem': 3,
                'simtime': 10000,
                'seed': 12,
                'neurons': 20}

# Neuron Parameter
neuron_params = {"C_m": 1.0,
                    "t_ref": 0.1,
                    "V_reset": 0.0,
                    "tau_m": 10.0,
                    "V_th": 2.0,
                    "E_L": 0.0}



mu, std, cv = experiment(exp_params, sim_params, neuron_params)


plt.plot(mu)
plt.ylim(0,100)
plt.show()

plt.plot(cv)
plt.ylim(0,5)
plt.show()








    # neuron_ts.append(n_ts)







# print(neuron_ts[0])
# print(neuron_ts[1])
# print(neuron_ts[2])
# print(neuron_ts[3])
# print(neuron_ts[4])


# %%



#     # Running Simulation
#     fr_theo, fr_sim, var_theo, var_sim, stds = run(sim_params, neuron_params)

# # Plotting
#     alpha = 0.7
#     fig, ax = plt.subplots()
#     ax.plot(stds, fr_theo, label='theory', color='k', alpha=alpha)
#     ax.plot(stds, fr_sim, label='simulation', color='r', alpha=alpha)
#     ax.set(ylabel='Firing Rate', xlabel='Voltage STD')
#     ax.legend()

#     # Firing Rate variance
#     fig, ax = plt.subplots()
#     ax.plot(stds, var_theo, label='theory', color='k', alpha=alpha)
#     ax.plot(stds, var_sim, label='simulation', color='red', alpha=alpha)
#     ax.set(ylabel='Variance', xlabel='Voltage STD')
#     ax.set_ylim(0, 5000)
#     ax.legend()

#     # Fano Factor
#     fig, ax = plt.subplots()
#     ax.plot(stds, np.sqrt(np.array(var_theo))*np.array(fr_theo), label='theory', color='k', alpha=alpha)
#     ax.plot(stds, np.sqrt(np.array(var_sim))*np.array(fr_sim), label='sim', color='red', alpha=alpha)
#     ax.set(ylabel='Coefficient of Variation', xlabel='Voltage STD')
#     ax.legend()

#     plt.show()
