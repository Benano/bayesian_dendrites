# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import nest
import scipy.integrate as integrate
from siegert import nu_0
import scipy

# %% Theory vs Simulation
def simulate_fr(sim_params, neuron_params):
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
    firing_rate = spikedetector.n_events/sim_params['simtime']*1000

    # Firing Rate
    return firing_rate

def theory_fr(sim_params,neuron_params):
    ''' Calculate the theoretical firing rate for a LIF neuron.'''

    if sim_params['theo'] == 'siegert':
        firing_rate = nu_0(neuron_params['tau_m'],neuron_params['t_ref'],
                           neuron_params['V_th'],neuron_params['V_reset'], sim_params['mean_mem'], sim_params['std_mem'])*1000


    elif sim_params['theo'] == 'brunel':
        ## Components
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
        firing_rate = 1/iti*1000

    return firing_rate

def theory_variance(sim_params,neuron_params):

    # Firing Rate
    fr = theory_fr(sim_params, neuron_params)

    # Bounds
    outer_bound_top = (neuron_params["V_th"] - sim_params["mean_mem"])/sim_params['std_mem']
    outer_bound_bottom = (neuron_params["V_reset"] - sim_params['mean_mem'])/sim_params['std_mem']

    # Functions
    f = lambda y, x: np.exp(x**2) * np.exp(y**2) * (1 + scipy.special.erf(y))**2

    # Integral
    integral = integrate.dblquad(f, outer_bound_bottom, outer_bound_top, lambda x: -5, lambda x: x)

    # Variance
    variance = fr**2 + 2 * np.pi * integral[0]

    return variance

def theory_vs_simulation(sim_params,neuron_params):
    '''Compare theory to simulation.'''

    # Simulation
    sim_fr = simulate_fr(sim_params, neuron_params)

    # Theoretical
    theo_fr = theory_fr(sim_params, neuron_params)
    theo_var = theory_variance(sim_params, neuron_params)

    return theo_fr, theo_var, sim_fr

def run_simulation(mem_std_range):
    fr_theo_rec = []
    fr_sim_rec = []
    var_theo_rec = []

    for mem_std in mem_std_range:
        print(mem_std)
        sim_params['std_mem'] = mem_std
        theo_fr, theo_var, sim_fr = theory_vs_simulation(sim_params,neuron_params)

        # Recording
        fr_theo_rec.append(theo_fr)
        var_theo_rec.append(theo_var)
        print(theo_var)
        fr_sim_rec.append(sim_fr)

    return fr_theo_rec, var_theo_rec, fr_sim_rec

if __name__ == "__main__":

    # Simulation Parameters
    sim_params = {'dt_noise': 0.1,
                'sim_res': 0.1,
                'mean_mem': -70,
                'std_mem': 33.0,
                'simtime': 200000,
                'seed': 7,
                'theo': 'siegert'}

    # Neuron Parameter
    neuron_params = {"C_m":1.0,
                    "t_ref":2.0,
                    "V_reset":-70.0,
                    "tau_m":20.0,
                    "V_th":-55.0}

    # Current input
    mem_stds = np.linspace(10,200,100)
    fr_theo_rec, var_theo_rec, fr_sim_rec = run_simulation(mem_stds)

    # %% Figure
    alpha = 0.7
    fig, ax  = plt.subplots()
    ax.plot(mem_stds, fr_theo_rec, label='theoretical', color='k',alpha=alpha)
    ax.plot(mem_stds, fr_sim_rec, label='simulation',color='r',alpha=alpha)
    ax.set(ylabel='Firing Rate',xlabel='Voltage std')
    ax.legend()

    alpha = 0.7
    fig, ax  = plt.subplots()
    ax.plot(mem_stds,var_theo_rec, label='theoretical', color='k',alpha=alpha)
    ax.set(ylabel='variance',xlabel='Voltage std')
    ax.legend()

    plt.show()
