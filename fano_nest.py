
"""
Running small tests on neurons.

Purpose :: Determining the parameters for the transition from firing
rate statistics to membrane voltage statistics

Method :: Simulation in a small example coded in nest.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import nest
import nest.voltage_trace
# - Parameters have to ve given as floats

# %% Simulating Single Neuron

# Kernel
nest.set_verbosity("M_WARNING")
nest.ResetKernel()

# Creating
neuron = nest.Create("iaf_psc_alpha")
vm = nest.Create("voltmeter")

# Setting Status
nest.SetStatus(neuron, "I_e", 376.0)
nest.SetStatus(vm, [{"withgid": True}])

# Connecting
nest.Connect(vm, neuron)

# Running
nest.Simulate(200.0)

# Voltage Trace
nest.voltage_trace.from_device(vm)
nest.voltage_trace.show()

# %% CUBA LIF Neuron
neuron = "COBA"
total_simtime = 2000.0
noise1_std = 7000.0
noise2_std = 2000.0

exp_nr = 500
bin_size = 100
samples = 200

step = int(total_simtime/samples)
result_mat = np.zeros((exp_nr, samples))

for exp in range(exp_nr):

    # Reset Kernel
    nest.ResetKernel()

    # Random Seed
    nest.SetKernelStatus({"rng_seeds": [exp]})
    np.random.RandomState(5)

    # Neuron
    if neuron == "CUBA":
        lif_neuron = nest.Create("iaf_psc_alpha")
    elif neuron == "COBA":
        lif_neuron = nest.Create("aeif_cond_alpha")

    # Noise
    noise = nest.Create("noise_generator")
    noise1 = nest.Create("noise_generator")

    # Configuring Noise
    nest.SetStatus(noise, {"start": 0.0, "stop": total_simtime/2,
                           "std": noise1_std})
    nest.SetStatus(noise1, {"start": total_simtime/2, "stop": total_simtime,
                            "std": noise2_std})

    # Multimeter
    multimeter = nest.Create("multimeter")
    nest.SetStatus(multimeter, {"withtime": True, "record_from": ["V_m"]})

    # Spike Detector
    spikedetector = nest.Create("spike_detector",
                                params={"withgid": True, "withtime": True})

    # Connections
    nest.Connect(noise, lif_neuron)
    nest.Connect(noise1, lif_neuron)
    nest.Connect(multimeter, lif_neuron)
    nest.Connect(lif_neuron, spikedetector)

    # Running
    nest.Simulate(2000.0)

    # Voltage Data
    dmm = nest.GetStatus(multimeter)[0]
    Vms = dmm["events"]["V_m"]
    ts = dmm["events"]["times"]

    # # Plotting Voltage
    # plt.figure(1)
    # plt.plot(ts, Vms)

    # Spike Data
    dSD = nest.GetStatus(spikedetector, keys="events")[0]
    evs = dSD["senders"]
    ts = dSD["times"]

    # # Plotting Spikes
    # plt.figure(2)
    # plt.plot(ts, evs, ".")
    # plt.show()

    # Results
    ts = ts.astype('int')

    for i in range(samples):
        slice_start = int((i*step))
        slice_end = int((i*step+bin_size))
        x = ts[ts < slice_end]
        y = x[x > slice_start]
        result_mat[exp, i] = len(y)

# Fano factor
mean = np.average(result_mat, axis=0)
var = np.var(result_mat, axis=0)
fano_factor = var/mean

# Plotting
plt.imshow(result_mat)
plt.show()

# Fano Factor
fig, ax = plt.subplots()
ax.plot(fano_factor)
ax.axhline(y=1, color='r')
ax.set_ylim(0, 4)
ax.set(title="Fano Factor", xlabel="Window", ylabel="Fano Factor")
fig.show()
fig.savefig('/Users/Benano/Desktop/Fano/Fanofactor.jpg', dpi=400)

# # Mean and Variance
# plt.Figure()
# plt.title("Mean & Variance")
# plt.plot(mean,label='mean')
# plt.plot(var,label="variance")
# plt.xlabel("window")
# plt.legend()
# plt.show()

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
