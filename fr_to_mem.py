# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import nest
import scipy.integrate as integrate
from siegert import nu_0

# %% Theory vs Simulation
def simulate_fr(sim_params, neuron_params):
    '''Simulate the firing rate of a neuron using the nest simulator.'''

    # Adjust I for dt_noise
    sim_params['std_I'] = sim_params['std_I']/sim_params['dt_noise']

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
    noise.set({"std":sim_params['std_I'],"dt":sim_params['dt_noise']})

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

    sim_params['std_I'] = sim_params['std_I']/sim_params['dt_noise']
    mean_mem = sim_params['mean_I']
    std_mem = (sim_params['std_I'])/neuron_params["C_m"] * np.sqrt(sim_params['dt_noise'] * neuron_params["tau_m"]/2)

    if sim_params['theo'] == 'siegert':
        firing_rate = nu_0(neuron_params['tau_m'],neuron_params['t_ref'],
                   neuron_params['V_th'],neuron_params['V_reset'], mean_mem, std_mem)*1000


    elif sim_params['theo'] == 'brunel':
        ## Components
        outer_bound_top = (neuron_params["V_th"] - mean_mem)/std_mem
        outer_bound_bottom = (neuron_params["V_reset"] - mean_mem)/std_mem
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

def theory_vs_simulation(sim_params,neuron_params):
    '''Compare theory to simulation.'''

    # Theoretical
    theo_fr = theory_fr(sim_params, neuron_params)

    # Simulation
    sim_fr = simulate_fr(sim_params, neuron_params)

    return theo_fr, sim_fr

def run_simulation(I_std_range):
    theo_rec = []
    sim_rec = []

    for I_std in I_std_range:
        print(I_std)
        sim_params['std_I'] = I_std
        theo_fr, sim_fr = theory_vs_simulation(sim_params,neuron_params)
        theo_rec.append(theo_fr)
        sim_rec.append(sim_fr)

    return theo_rec, sim_rec

if __name__ == "__main__":

    # Simulation Parameters
    sim_params = {'dt_noise': 0.1,
                'sim_res': 0.1,
                'mean_I': -70,
                'std_I': 33.0,
                'simtime': 80000,
                'seed': 7,
                'theo': 'siegert'}

    # Neuron Parameter
    neuron_params = {"C_m":1.0,
                    "t_ref":2.0,
                    "V_reset":-70.0,
                    "tau_m":20.0,
                    "V_th":-55.0}

    # Current input
    I_stds = np.linspace(0,50,50)
    theo_rec, sim_rec = run_simulation(I_stds)

    # %% Figure
    alpha = 0.7
    fig, ax  = plt.subplots()
    ax.plot(I_stds,theo_rec,label='theoretical',color='k',alpha=alpha)
    ax.plot(I_stds, sim_rec,label='simulation',color='r',alpha=alpha)
    ax.set(ylabel='Firing Rate',xlabel='Current std (herpes)')
    ax.legend()
    plt.show()
