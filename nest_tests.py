# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import nest
import nest.voltage_trace
import scipy as sc
import statistics as stats

# # %% Simulating Single Neuron
# # Kernel
nest.set_verbosity("M_WARNING")
# %% Functions

def find_slope(neuron_params):

    # Input variances
    I_std = np.linspace(0.0,1.0,10)

    for i in I_std:
        nest.ResetKernel()

        # Random Seed
        # nest.SetKernelStatus({"rng_seeds" : [3]})
        np.random.RandomState(5)

        # Neuron
        lif_neuron = nest.Create("iaf_psc_alpha")
        nest.SetStatus(lif_neuron,{"V_th": 10000000.0})
        nest.SetStatus(lif_neuron,neuron_params)

        # Noise
        noise = nest.Create("noise_generator")

        # Configuring Noise
        nest.SetStatus(noise,{"start":0.0,"stop":1000.0,"std":i})

        # Multimeter
        multimeter = nest.Create("multimeter")
        nest.SetStatus(multimeter, {"record_from":["V_m"]})

        # Connections
        nest.Connect(noise, lif_neuron)
        nest.Connect(multimeter, lif_neuron)

        # Running
        nest.Simulate(1000.0)

        # Voltage Data
        dmm = nest.GetStatus(multimeter)[0]
        Vms = dmm["events"]["V_m"]

        # Variance
        var_vec.append(np.var(Vms))
        std_vec.append(stats.stdev(Vms))

    # Slope
    y1 = std_vec[1]
    y2 = std_vec[-1]
    x1 = I_std[1]
    x2 = I_std[-1]

    slope = (y2-y1)/(x2-x1) 
    print(slope)

    return slope

# Parameter

# Neuron Parameters
neuron_params = {"C_m":1.0,
                 "t_ref":2.0,
                 "V_reset":10.0,
                 "I_e":0.0,
                 "tau_m":10.0,
                 "V_th":20.0}

# Mean and variance I
mean_I = 0.0
std_I = 1.0

# mean and variance membrane
mean_mem = mean_I
std_mem = find_slope(neuron_params) * std_I

# Boundaries
outer_bound_top = (neuron_params["V_th"] - mean_mem)/std_mem
outer_bound_bottom = (neuron_params["V_reset"] - mean_mem)/std_mem

# Function
outer = lambda x: np.exp(x**2)
inner = lambda y: np.exp(-y**2)

tau_ref = neuron_params["t_ref"]
tau_m = neuron_params["tau_m"]

full = lambda y,x: tau_ref + 2 * tau_m * outer(x) * inner(y)
# %%
# sc.integrate.dblquad(full,outer_bound_bottom,outer_bound_top,lambda x: -np.inf, lambda x: 1.0)
