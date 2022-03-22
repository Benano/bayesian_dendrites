# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import nest
import nest.voltage_trace
import scipy as sc
import statistics as stats

# # %% Simulating Single Neuron
# # Kernel
# nest.set_verbosity("M_WARNING")
# nest.ResetKernel()

# # Creating
# neuron = nest.Create("iaf_psc_alpha")
# vm = nest.Create("voltmeter")

# # Setting Status
# nest.SetStatus(neuron,"I_e",376.0)
# # nest.SetStatus(vm,[{"withgid":True}])

# # Connecting
# nest.Connect(vm,neuron)

# # Running
# nest.Simulate(200.0)

# # Voltage Trace
# nest.voltage_trace.from_device(vm)
# nest.voltage_trace.show()

# # %% CUBA LIF Neuron
# neuron = "CUBA"
# total_simtime = 2000.0
# noise1_std = 20.0
# noise2_std = 2.0

# exp_nr = 100
# bin_size = 50
# samples = 100

# step = int(total_simtime/samples)
# result_mat = np.zeros((exp_nr,samples))
# spike_plot = np.zeros((exp_nr,int(total_simtime)))

# neuron_params = {"C_m":1.0,
#                 "t_ref":2.0,
#                 "V_reset":-70.0,
#                 "V_th": -60.0,
#                 "I_e":3.0,
#                 "tau_m":20.0}

# for exp in range(exp_nr):

#     # Reset Kernel
#     nest.ResetKernel()

#     # Random Seed
#     nest.SetKernelStatus({"rng_seeds" : [exp]})
#     np.random.RandomState(5)

#     # Neuron
#     if neuron == "CUBA":
#         lif_neuron = nest.Create("iaf_psc_alpha")
#     elif neuron == "COBA":
#         lif_neuron = nest.Create("aeif_cond_alpha")

#     nest.SetStatus(lif_neuron,neuron_params)

#     # Noise
#     noise = nest.Create("noise_generator")
#     noise1 = nest.Create("noise_generator")

#     # Configuring Noise
#     nest.SetStatus(noise,{"start":0.0,"stop":total_simtime/2,"std":noise1_std})
#     nest.SetStatus(noise1,{"start":total_simtime/2,"stop":total_simtime,"std":noise2_std})

#     # Multimeter
#     multimeter = nest.Create("multimeter")
#     nest.SetStatus(multimeter, {"withtime":True, "record_from":["V_m"]})

#     # Spike Detector
#     spikedetector = nest.Create("spike_detector",
#                     params={"withgid": True, "withtime": True})

#     # Connections
#     nest.Connect(noise, lif_neuron)
#     nest.Connect(noise1, lif_neuron)
#     nest.Connect(multimeter, lif_neuron)
#     nest.Connect(lif_neuron, spikedetector)

#     # Running
#     nest.Simulate(total_simtime)

#     # Voltage Data
#     dmm = nest.GetStatus(multimeter)[0]
#     Vms = dmm["events"]["V_m"]
#     ts = dmm["events"]["times"]

#     # # Plotting Voltage
#     # plt.figure(1)
#     # plt.plot(ts, Vms)

#     # Spike Data
#     dSD = nest.GetStatus(spikedetector,keys="events")[0]
#     evs = dSD["senders"]
#     ts = dSD["times"]
#     ts = ts.astype('int')

#     # spike_plot[exp][ts] = 1

#     # # Plotting Spikes
#     # plt.figure(2)
#     # plt.plot(ts, evs, ".")
#     # plt.show()

#     # Results
#     for i in range(samples):
#         slice_start = int((i*step))
#         slice_end = int((i*step+bin_size))
#         x = ts[ts<slice_end]
#         y = x[x>slice_start]
#         result_mat[exp,i] = len(y)


# # Fano factor
# mean = np.average(result_mat,axis=0)
# var = np.var(result_mat,axis=0)
# fano_factor = var/mean
# # %%

# # # Spikes
# # fig,ax = plt.subplots(dpi=800)
# # ax.imshow(spike_plot,cmap='binary')
# # ax.set_aspect(6)
# # fig.savefig("/Users/Benano/Desktop/spikes.png")


# # %%

# # %%
# # Windows
# fig,ax = plt.subplots(dpi=500)
# ax.imshow(result_mat)
# ax.set_aspect(2)
# fig.savefig("/Users/Benano/Desktop/test.png")
# plt.show()

# # %%
# # Fano Factor
# fig, ax = plt.subplots(dpi=500)
# ax.plot(fano_factor)
# ax.axhline(y=1,color='r')
# ax.set_ylim(0,4)
# ax.set(title="Fano Factor",xlabel="Window",ylabel="Fano Factor")
# fig.show()


# # # Mean and Variance
# # plt.Figure()
# # plt.title("Mean & Variance")
# # plt.plot(mean,label='mean')
# # plt.plot(var,label="variance")
# # plt.xlabel("window")
# # plt.legend()
# # plt.show()


# %% Testing variance of noisy input
# Simulation Parameters
# total_simtime = 1000.0
# I_std = np.linspace(0.0,1.0,10)
# tau_m_vec = np.linspace(1,1000,50)

# # Neuron parameters
# neuron_params = {"C_m":1.0,
#                  "t_ref":2.0,
#                  "V_reset":20.0,
#                  "I_e":0.0,
#                  "tau_m":10.0,
#                  "V_th":10000000.0}
# std_vec = []
# var_vec = []
# slope_vec = []


# # %%
# # Reset Kernel
# for k in tau_m_vec:
#     neuron_params["tau_m"] = k
#     for en,i in enumerate(I_std):
#         nest.ResetKernel()

#         # Random Seed
#         # nest.SetKernelStatus({"rng_seeds" : [3]})
#         np.random.RandomState(5)

#         # Neuron
#         lif_neuron = nest.Create("iaf_psc_alpha")
#         lif_neuron.set(neuron_params)
#         # nest.set
#         # nest.SetStatus(lif_neuron,{"V_th": 10000000.0})
#         # nest.SetStatus(lif_neuron,neuron_params)
#         # a = nest.GetStatus(lif_neuron,neuron_params)

#         # Noise
#         noise = nest.Create("noise_generator")

#         # Configuring Noise
#         noise.set({"start":0.0,"stop":total_simtime,"std":I_std[en]})

#         # Multimeter
#         multimeter = nest.Create("multimeter")
#         multimeter.set({"record_from":["V_m"]})

#         # Spike Detector
#         spikedetector = nest.Create("spike_recorder")


#         # Connections
#         nest.Connect(noise, lif_neuron)
#         nest.Connect(multimeter, lif_neuron)
#         nest.Connect(lif_neuron, spikedetector)

#         # Running
#         a = nest.Simulate(total_simtime)

#         # Voltage Data
#         dmm = nest.GetStatus(multimeter)[0]
#         Vms = dmm["events"]["V_m"]
#         ts = dmm["events"]["times"]

#         # Variance
#         var_vec.append(np.var(Vms))
#         std_vec.append(stats.stdev(Vms))

#     # Slope
#     y1 = std_vec[1]
#     y2 = std_vec[-1]
#     x1 = I_std[1]
#     x2 = I_std[-1]

#     slope = (y2-y1)/(x2-x1)
#     slope_vec.append(slope)

# # %%Plotting
# plt.Figure()
# plt.title("Input SD vs Membrane SD")
# plt.plot(I_std,std_vec)
# plt.xlabel("std pA")
# plt.ylabel("std mV")

# # %%
# plt.Figure()
# plt.title("Input SD vs Membrane VAR")
# plt.plot(I_std,var_vec)
# plt.xlabel("std pA")
# plt.ylabel("var mV")

# # %%
# plt.Figure()
# plt.plot(Vms)

# # %% Slope
# y1 = std_vec[1]
# y2 = std_vec[-1]
# x1 = I_std[1]
# x2 = I_std[-1]

# slope = (y2-y1)/(x2-x1)
# print("Slope: "+ str(slope))

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
sc.integrate.dblquad(full,outer_bound_bottom,outer_bound_top,lambda x: -np.inf, lambda x: 1.0)


# %% Toy Example



















# %%

# %%
