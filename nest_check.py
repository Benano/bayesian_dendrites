#!/usr/bin/env python3
# Imports
import nest
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats


# Turn off Errors
nest.set_verbosity("M_WARNING")

# Running Neuron
# %% Testing variance of noisy input
# Simulation Parameters
total_simtime = 1000.0
I_std = np.linspace(0.0,5.0,20)
# tau_m_vec = np.linspace(1,1000,50)

# Neuron parameters
neuron_params = {"C_m":1.0,
                 "t_ref":2.0,
                 "V_reset":20.0,
                 "I_e":0.0,
                 "tau_m":10.0,
                 "V_th":10000000.0}
std_vec = []
var_vec = []
slope_vec = []

# Reset Kernel
for k in tau_m_vec:
    neuron_params["tau_m"] = k

    for en,i in enumerate(I_std):
        nest.ResetKernel()

        # Random Seed
        # nest.SetKernelStatus({"rng_seeds" : [3]})
        np.random.RandomState(5)

        # Neuron
        lif_neuron = nest.Create("iaf_psc_alpha")
        lif_neuron.set(neuron_params)
        # nest.set
        # nest.SetStatus(lif_neuron,{"V_th": 10000000.0})
        # nest.SetStatus(lif_neuron,neuron_params)
        # a = nest.GetStatus(lif_neuron,neuron_params)

        # Noise
        noise = nest.Create("noise_generator")

        # Configuring Noise
        noise.set({"start":0.0,"stop":total_simtime,"std":I_std[en]})

        # Multimeter
        multimeter = nest.Create("multimeter")
        multimeter.set({"record_from":["V_m"]})

        # Spike Detector
        spikedetector = nest.Create("spike_recorder")


        # Connections
        nest.Connect(noise, lif_neuron)
        nest.Connect(multimeter, lif_neuron)
        nest.Connect(lif_neuron, spikedetector)

        # Running
        a = nest.Simulate(total_simtime)

        # Voltage Data
        dmm = nest.GetStatus(multimeter)[0]
        Vms = dmm["events"]["V_m"]
        ts = dmm["events"]["times"]

        # Variance
        var_vec.append(np.var(Vms))
        std_vec.append(stats.stdev(Vms))

    # Slope
    y1 = std_vec[1]
    y2 = std_vec[-1]
    x1 = I_std[1]
    x2 = I_std[-1]

    slope = (y2-y1)/(x2-x1)
    slope_vec.append(slope)

# Slopes
# plt.Figure()
# plt.plot(slope_vec)
# plt.show()

# %%Plotting
plt.Figure()
plt.title("Input SD vs Membrane SD")
plt.plot(I_std,std_vec)
plt.xlabel("std pA")
plt.ylabel("std mV")
plt.show()

# %%
plt.Figure()
plt.title("Input SD vs Membrane VAR")
plt.plot(I_std,var_vec)
plt.xlabel("std pA")
plt.ylabel("var mV")
plt.show()

# %% Hello
plt.Figure()
plt.plot(Vms)
plt.show()

# %% Slope
y1 = std_vec[1]
y2 = std_vec[-1]
x1 = I_std[1]
x2 = I_std[-1]

slope = (y2-y1)/(x2-x1)
print("Slope: "+ str(slope))
