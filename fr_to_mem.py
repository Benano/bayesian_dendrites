# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import nest
import scipy.integrate as integrate
from siegert import nu_0
import scipy

# %% Theory vs Simulation
def simulate(sim_params, neuron_params):
    '''Simulate the firing rate of a neuron using the nest simulator.'''

    # dt noise correction
    # Sigma = np.sqrt(2/(sim_params['dt_noise']*neuron_params['tau_m']))/sim_params['std_mem']
    Sigma = np.sqrt(2/(sim_params['dt_noise']*neuron_params['tau_m'])) * neuron_params['C_m'] * sim_params['std_mem']
    print('Sigma: ' + str(Sigma))

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
    noise.set({"std":Sigma,"dt":sim_params['dt_noise']})

    # Spike Detector
    spikedetector = nest.Create("spike_recorder")

    # Connections
    nest.Connect(noise, lif_neuron)
    nest.Connect(lif_neuron, spikedetector)

    # Running
    nest.Simulate(sim_params['simtime'])

    # Firing Rate
    simulated_firing_rate = spikedetector.n_events/sim_params['simtime']*1000

    # Spike Data
    dSD = nest.GetStatus(spikedetector,keys="events")[0]
    evs = dSD["senders"]
    ts = dSD["times"]
    ts = ts.astype('int')

    # Variance
    simulated_variance = np.var(np.diff(ts))

    # Firing Rate
    return simulated_firing_rate, simulated_variance, ts

def theorize(sim_params,neuron_params):
    ''' Calculate the theoretical firing rate and variance for a LIF neuron.'''

    # %% Firing Rate
    outer_bound_top = (neuron_params["V_th"] - sim_params["mean_mem"])/sim_params['std_mem']
    outer_bound_bottom = (neuron_params["V_reset"] - sim_params['mean_mem'])/sim_params['std_mem']
    outer = lambda x: np.exp(x**2)
    inner = lambda y: np.exp(-y**2)
    tau_ref = neuron_params["t_ref"]
    tau_m = neuron_params["tau_m"]

    # Function - Equation 21 from Brunel 2000
    f = lambda x: np.exp(x**2) * (1 + scipy.special.erf(x))
    integral = integrate.quad(f, outer_bound_bottom, outer_bound_top)
    iti = tau_ref + tau_m * np.sqrt(np.pi) * integral[0]
    theory_firing_rate = 1/iti*1000

    # %% Variance

    # Function
    f = lambda y, x: np.exp(x**2) * np.exp(y**2) * (1 + scipy.special.erf(y))**2
    integral = integrate.dblquad(f, outer_bound_bottom, outer_bound_top, lambda x: -5, lambda x: x)
    theory_variance = theory_firing_rate**2 + 2 * np.pi * integral[0]

    return theory_firing_rate, theory_variance

def run_simulation(mem_std_range):
    fr_theo_rec = []
    fr_sim_rec = []
    var_theo_rec = []
    var_sim_rec = []

    for mem_std in mem_std_range:

        # Setting membrane std
        sim_params['std_mem'] = mem_std

        # Theory
        theo_fr, theo_var = theorize(sim_params, neuron_params)

        # Simulation
        sim_fr, sim_var, ts = simulate(sim_params, neuron_params)

        # Recording
        fr_theo_rec.append(theo_fr)
        fr_sim_rec.append(sim_fr)
        var_theo_rec.append(theo_var)
        var_sim_rec.append(sim_var)

    return fr_theo_rec, fr_sim_rec, var_theo_rec, var_sim_rec

if __name__ == "__main__":

    # Simulation Parameters
    sim_params = {'dt_noise': 0.1,
                'sim_res': 0.1,
                'mean_mem': -70,
                'std_mem': 10.0,
                'simtime': 100000,
                'seed': 7,
                'theo': 'siegert'}

    # Neuron Parameter
    neuron_params = {"C_m": 1.0,
                    "t_ref": 2.0,
                    "V_reset": -70.0,
                    "tau_m": 20.0,
                    "V_th": -55.0}

    # Voltage input
    mem_stds = np.linspace(6, 25, 100)

    # Running Simulation
    fr_theo_rec, fr_sim_rec, var_theo_rec, var_sim_rec = run_simulation(mem_stds)

    # Plotting
    alpha = 0.7
    fig, ax  = plt.subplots()
    ax.plot(mem_stds, fr_theo_rec, label='theory', color='k',alpha=alpha)
    ax.plot(mem_stds, fr_sim_rec, label='simulation',color='r',alpha=alpha)
    ax.set(ylabel='Firing Rate',xlabel='Voltage STD')
    ax.legend()

    alpha = 0.7
    fig, ax  = plt.subplots()
    ax.plot(mem_stds, var_theo_rec, label='theory', color='k', alpha=alpha)
    ax.plot(mem_stds, var_sim_rec, label='simulation', color='red', alpha=alpha)
    ax.set(ylabel='Variance', xlabel='Voltage STD')
    ax.legend()

    plt.show()

    # # Look at single ts
    # fr, var, ts = simulate(sim_params, neuron_params)
    # iti_var = np.diff(ts)
    # plt.hist(iti_var,bins=100)
    # plt.show()
