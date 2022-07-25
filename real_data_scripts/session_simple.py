# %% Testing


import os
import numpy as np
import pandas as pd
import scipy
import neo
import elephant
import quantities as pq
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from constants import *
from loadmat import loadmat
import mat73

from

"""
The data paths are set up in constants.py, make sure to edit there
"""


class Session():

    def __init__(self, animal_id='', session_id='', dataset='',
                 verbose=1, subfolder='data'):

        # print('\nInitializing Session object for: \n- animal ID: {}'
        #       '\n- Session ID: {}\n'.format(animal_id, session_id))
        self.data_folder = os.path.join(DATA_FOLDER, subfolder, dataset, animal_id, session_id)
        self.verbose = verbose
        self.dataset = dataset
        self.animal_id = animal_id
        self.session_id = session_id

    def load_data(self, load_spikes=True, load_lfp=False,
                  load_lfp_lazy=True, use_newtrials=True, load_video=False):

        """
        :param load_spikes: if True, loads spike times
        :param load_lfp: if True, loads the LFP data
        :param load_lfp_lazy: if True, loads the lfpData_lazy.mat which contains metdata
        about the lfp without actually loading the lfp signals themselves
        :param use_newtrials: if True, loads trialData_newtrials.mat, which
        contains catch trials defined a posteriori
        :param load_video: if True, loads video data
        """

        self.session_data_path = os.path.join(self.data_folder, 'sessionData.mat')

        # if self.dataset.casefold() == 'VisOnlyTwolevels'.casefold():
        #     print('Loading alternative trial definition')
        #     self.trial_data_path = os.path.join(self.data_folder, 'trialData_newtrials.mat')
        #     trial_data_key = 'trialData_newtrials'
        # else:
        #     self.trial_data_path = os.path.join(self.data_folder, 'trialData.mat')
        #     trial_data_key = 'trialData'

        if use_newtrials:
            self.trial_data_path = os.path.join(self.data_folder, 'trialData_newtrials.mat')
        else:
            self.trial_data_path = os.path.join(self.data_folder, 'trialData.mat')

        trial_data_key = 'trialData'

        self.lfp_data_path = os.path.join(self.data_folder, 'lfpData.mat')
        self.lfp_lazy_data_path = os.path.join(self.data_folder, 'lfpData_lazy.mat')
        self.spike_data_path = os.path.join(self.data_folder, 'spikeData.mat')
        self.video_data_path = os.path.join(self.data_folder, 'videoData.mat')

        if self.verbose > 1:
            print('Loading session data at {}'.format(self.session_data_path))
        session_data = loadmat(self.session_data_path)['sessionData']

        if self.verbose > 1:
            print('Loading trial data at {}'.format(self.trial_data_path))
        trial_data = loadmat(self.trial_data_path)[trial_data_key]

        if load_lfp:
            if self.verbose > 1 :
                print('Loading lfp data at {}'.format(self.lfp_data_path))
            lfp_data = loadmat(self.lfp_data_path)['lfpData']
            self.lfp_signal_loaded = True
        elif load_lfp_lazy:
            try:
                if self.verbose > 1 :
                    print('Loading lfp data [lazy loading] at {}'.format(self.lfp_lazy_data_path))
                lfp_data = loadmat(self.lfp_lazy_data_path)['lfpData']
                self.lfp_signal_loaded = False
            except FileNotFoundError:
                print('Could not find lazy LFP data')
                lfp_data = None
        else:
            lfp_data = None

        if load_spikes:
            if self.verbose > 1 :
                print('Loading spike data at {}'.format(self.spike_data_path))
            spike_data = loadmat(self.spike_data_path)['spikeData']
        else:
            spike_data = None

        if load_video:
            video_data = mat73.loadmat(self.video_data_path)['videoData']
        else:
            video_data = None

        self._initialize(session_data=session_data,
                         trial_data=trial_data,
                         spike_data=spike_data,
                         lfp_data=lfp_data,
                         video_data=video_data)


    def _initialize(self, session_data, trial_data, spike_data=None, lfp_data=None,
                    center_lfp=True, video_data=None):

        self.session_data = session_data

        # --- ADD TRIAL DATA ---
        trial_data_df = pd.DataFrame(columns=trial_data.keys())
        for key in trial_data.keys():
            trial_data_df[key] = trial_data[key]
        self.trial_data = trial_data_df

        self.trial_data['responseSide'] = [i if isinstance(i, str) else 'n' for i in
                                           self.trial_data.responseSide]

        responseModality = []
        for s in self.trial_data['responseSide'] :
            if s == 'L' :
                r = 'Y'
            elif s == 'R' :
                r = 'X'
            elif s == 'n' :
                r = 'n'
            responseModality.append(r)
        self.trial_data['responseModality'] = responseModality

        # if self.dataset.casefold() == 'VisOnlyTwolevels'.casefold():
        #     print('Generating artificial trial numbers for VisOnlyTwolevels data')
        #     audiot = self.trial_data[self.trial_data['trialType'] == 'Y']
        #     assert np.isnan(audiot['trialNum']).sum() == audiot.shape[0]
        #     self.trial_data.loc[self.trial_data['trialType'] == 'Y', 'trialNum'] = np.arange(5000, 5000+audiot.shape[0])
        #     self.trial_data['trialNum'] = self.trial_data['trialNum'].astype(int)

        first_lick_times = []
        for times in self.trial_data['lickTime']:
            try:
                first_lick_time = times[0]
            except IndexError:
                first_lick_time = np.nan
            except TypeError:
                first_lick_time = times
            first_lick_times.append(first_lick_time)

        self.trial_data['firstlickTime'] = first_lick_times
        self.trial_data['Lick'] = [0 if n == 1 else 1 for n in self.trial_data['noResponse']]


        if np.isin(self.animal_id, changed_audio_presentation_animals) :

            #print('Fixing labels of octave sessions')
            self.trial_data = self.trial_data.drop(labels=['audioFreqChangeNorm',
                                                          'audioFreqPostChangeNorm'], axis=1)
            self.trial_data = self.trial_data.rename(columns={"audioOctChangeNorm" : "audioFreqChangeNorm",
                                                              "audioOctPostChangeNorm" : "audioFreqPostChangeNorm"})


        # RUN CHECKS ON TRIAL DATA
        # assert set(self.trial_data['audioFreqPostChangeNorm'].unique()) == set([1, 2, 3, 4])
        # assert set(self.trial_data['audioFreqChangeNorm'].unique()) == set([1, 2, 3])

        #assert set(self.trial_data['visualOriPostChangeNorm'].unique()) == set([1, 2, 3, 4])

        # --- ADD SPIKE DATA ---
        self.spike_data = spike_data
        if spike_data is not None:
            self.spike_time_stamps = spike_data['ts']
            # since these are originally matlab indices, turn to string to avoid
            # that they are used here as indices
            self.spike_data['ch'] = self.spike_data['ch'].astype(str)
            self.session_t_start = np.hstack(self.spike_data['ts']).min() * pq.us
            self.session_t_stop = np.hstack(self.spike_data['ts']).max() * pq.us

            # make sure there are no spaces in the cell ids
            self.spike_data['cell_ID'] = np.array([s.replace(" ", "") for s in self.spike_data['cell_ID']])

            if np.isin(self.session_id, doubleshank_sessions):
                self.spike_data['cell_ID'] = np.array([s+r for s, r in zip(self.spike_data['cell_ID'],
                                                                           self.spike_data['shankIdx'].astype(str))])

            self.cell_id = self.spike_data['cell_ID'].astype(int).astype(str)
            # RUN CHECKS
            # no duplicate cell ids
            assert pd.value_counts(self.spike_data['cell_ID']).max() == 1

        # --- ADD LFP DATA ---
        self.lfp_data = lfp_data

        if lfp_data is not None:
            self.sampling_rate = lfp_data['fs'][0]

            self.session_t_start = lfp_data['t_start'][0] * pq.us
            self.session_t_stop = lfp_data['t_end'][0] * pq.us
            self.delta_t = (1 / self.sampling_rate) * 1e6 * pq.us

            self.channel_id = lfp_data['channel_ID'].astype(str)
            # self.lfp_times = np.arange(self.session_t_start,
            #                            self.session_t_stop+self.delta_t, self.delta_t)

            if self.lfp_signal_loaded:
                self.lfp_data['signal'] = np.vstack(self.lfp_data['signal'])

                if center_lfp:
                    self.lfp_data['signal'] = scipy.signal.detrend(self.lfp_data['signal'],
                                                                    type='constant')

                self.lfp_times = self.session_t_start + np.arange(self.lfp_data['signal'].shape[1]) / (
                                             self.sampling_rate * pq.Hz)

            # RUN CHECKS ON LFP DATA
            assert np.all(lfp_data['fs']==self.sampling_rate)
            assert np.all(lfp_data['butter_applied']==1)
            assert np.all(lfp_data['kaiser_applied'] == 0)
            assert np.all(lfp_data['t_units']=='us')
            assert np.all(lfp_data['signal_units']=='V')
            assert np.all(lfp_data['t_start']==lfp_data['t_start'][0])
            assert np.all(lfp_data['t_end']==lfp_data['t_end'][0])
            if self.lfp_signal_loaded:
                assert self.lfp_times.shape[0] == self.lfp_data['signal'].shape[1]

        if video_data is not None:
            self.video_data = video_data


    # --- TRIAL SELECTION AND METADATA ---

    def select_trials(self, trial_type=None, only_correct=False,
                      visual_post_norm=None, audio_post_norm=None,
                      visual_pre_norm=None, audio_pre_norm=None,
                      visual_change=None, auditory_change=None, exclude_last=20,
                      response=None, vec_response=None,
                      response_side=None, filter_responseModality=False):

        """
        Obtain a list of trial numbers with specific requirements

        ---

        trial_type can be just a string, or a list of strings if you
        want to include more trial types.


        vecResponse=1 is auditory, vecResponse=2 is visual,
        vecResponse=3 is noresponse

        auditory_pre and auditory_post are the pre and post split frequencies.
        auditory_pre=None, auditory_post=None removed because I always
        use the norm version, and this can generate confusion
        """

        total_n_trials = self.trial_data.shape[0]
        selected_trials = []

        if trial_type is not None:
            if isinstance(trial_type, str):
                trial_type = [trial_type]
            mask = np.isin(self.trial_data['trialType'], trial_type)
            sel_trials = self.trial_data.loc[mask, 'trialNum'].tolist()
            selected_trials.append(sel_trials)

        if response is not None:
            if isinstance(response, int):
                response = [response]
            mask = np.isin(self.trial_data['correctResponse'], response)
            sel_trials = self.trial_data.loc[mask, 'trialNum'].tolist()
            selected_trials.append(sel_trials)

        if vec_response is not None:
            if isinstance(response, int):
                vec_response = [vec_response]
            mask = np.isin(self.trial_data['vecResponse'], vec_response)
            sel_trials = self.trial_data.loc[mask, 'trialNum'].tolist()
            selected_trials.append(sel_trials)

        if only_correct:
            ind = self.trial_data['correctResponse'] == 1
            sel_trials = self.trial_data.loc[ind, 'trialNum'].tolist()
            selected_trials.append(sel_trials)

        if filter_responseModality:
            mask = np.isin(self.trial_data['responseModality'], ['X', 'Y'])
            sel_trials = self.trial_data.loc[mask, 'trialNum'].tolist()
            selected_trials.append(sel_trials)

        if visual_change is not None:
            if isinstance(visual_change, (float, int)):
                visual_change = [visual_change]
            mask = np.isin(self.trial_data['visualOriChangeNorm'], visual_change)
            sel_trials = self.trial_data.loc[mask, 'trialNum'].tolist()
            selected_trials.append(sel_trials)

        if auditory_change is not None:
            if isinstance(auditory_change, (float, int)):
                auditory_change = [auditory_change]
            mask = np.isin(self.trial_data['audioFreqChangeNorm'], auditory_change)
            sel_trials = self.trial_data.loc[mask, 'trialNum'].tolist()
            selected_trials.append(sel_trials)

        # if auditory_pre is not None:
        #     ind = self.trial_data['audioFreqPreChange'] == auditory_pre
        #     sel_trials = self.trial_data.loc[ind, 'trialNum'].tolist()
        #     selected_trials.append(sel_trials)
        #
        # if auditory_post is not None:
        #     if isinstance(auditory_post, (float, int)):
        #         auditory_post = [auditory_post]
        #     mask = np.isin(self.trial_data['audioFreqPostChange'], auditory_post)
        #     sel_trials = self.trial_data.loc[mask, 'trialNum'].tolist()
        #     selected_trials.append(sel_trials)

        if audio_pre_norm is not None:
            if isinstance(audio_pre_norm, (float, int)):
                audio_pre_norm = [audio_pre_norm]
            mask = np.isin(self.trial_data['audioFreqPreChangeNorm'], audio_pre_norm)
            sel_trials = self.trial_data.loc[mask, 'trialNum'].tolist()
            selected_trials.append(sel_trials)

        if audio_post_norm is not None:
            if isinstance(audio_post_norm, (float, int)):
                audio_post_norm = [audio_post_norm]
            mask = np.isin(self.trial_data['audioFreqPostChangeNorm'], audio_post_norm)
            sel_trials = self.trial_data.loc[mask, 'trialNum'].tolist()
            selected_trials.append(sel_trials)

        if visual_pre_norm is not None:
            if isinstance(visual_pre_norm, (float, int)):
                visual_pre_norm = [visual_pre_norm]
            mask = np.isin(self.trial_data['visualOriPreChangeNorm'], visual_pre_norm)
            sel_trials = self.trial_data.loc[mask, 'trialNum'].tolist()
            selected_trials.append(sel_trials)

        if visual_post_norm is not None:
            if isinstance(visual_post_norm, (float, int)):
                visual_post_norm = [visual_post_norm]
            mask = np.isin(self.trial_data['visualOriPostChangeNorm'], visual_post_norm)
            sel_trials = self.trial_data.loc[mask, 'trialNum'].tolist()
            selected_trials.append(sel_trials)

        if response_side is not None:
            ind = self.trial_data['responseSide'] == response_side
            sel_trials = self.trial_data.loc[ind, 'trialNum'].tolist()
            selected_trials.append(sel_trials)

        if len(selected_trials) > 0:
            selected_final = list(set.intersection(*map(set, selected_trials)))
        else:
            selected_final = self.trial_data['trialNum'].tolist()

        perc = 100*(len(selected_final) / total_n_trials)
        if self.verbose > 1:
            print('> Finished trial selection'
                  '\n ---> {} out of {} trials selected '
                  '({:.1f}%)'.format(len(selected_final), total_n_trials, perc))

        if exclude_last is not None:
            if self.verbose > 1 :
                print('Excluding last {} trial of the session from the selected ones'.format(exclude_last))
            last_trials = self.trial_data['trialNum'][-exclude_last:].tolist()
            selected_final = list(set(selected_final)-set(last_trials))

        selected_final.sort()

        return selected_final


    def get_trial_info(self, trial_number):
        return self.trial_data[self.trial_data['trialNum'] == trial_number]

    def get_trials_info(self, trial_numbers):
        ind = np.where(np.isin(self.trial_data['trialNum'], trial_numbers))[0]
        out = self.trial_data.iloc[ind]
        return out

    def get_type_of_trial(self, trial_number):
        trial_info = self.get_trial_info(trial_number)
        return trial_info['trialType'].iloc[0]

    def get_response_side_of_trial(self, trial_number):
        trial_info = self.get_trial_info(trial_number)
        return trial_info['responseSide'].iloc[0]

    def get_correct_response_of_trial(self, trial_number):
        trial_info = self.get_trial_info(trial_number)
        return trial_info['correctResponse'].iloc[0]

    def get_stimulus_change_of_trial(self, trial_number):
        trial_info = self.get_trial_info(trial_number)
        trial_type = self.get_type_of_trial(trial_number)
        if trial_type == 'X':
            stim = trial_info['visualOriChangeNorm'].iloc[0]
        elif trial_type == 'Y':
            stim = trial_info['audioFreqChangeNorm'].iloc[0]
        elif trial_type == 'P':
            stim = 0
        else:
            raise ValueError('Stimulus identity for which modality?')
        return stim

    def get_lick_time_of_trial(self, trial_number):
        trial_info = self.get_trial_info(trial_number)
        return trial_info['firstlickTime'].iloc[0]

    def get_lick_time_of_trials(self, trial_numbers):
        return [self.get_lick_time_of_trial(t) for t in trial_numbers]

    def get_stimulus_identity_of_trial(self, trial_number):
        trial_info = self.get_trial_info(trial_number)
        trial_type = self.get_type_of_trial(trial_number)
        if trial_type == 'X':
            stim = trial_info['visualOriPostChangeNorm'].iloc[0]
        elif trial_type == 'Y':
            stim = trial_info['audioFreqPostChangeNorm'].iloc[0]
        elif trial_type == 'P':
            stim = 0
        else:
            raise ValueError('Stimulus identity for which modality?')
        return stim

    def make_target(self, trial_numbers, target_name,
                    n_time_bins_per_trial=None, coarse=True):

        """
        Return a target variable for the select trials. If n_time_per_bins
        is passed, the target is repeated so that for every time point
        you get a target value.
        """
        sel_trial_ind = np.isin(self.trial_data['trialNum'], trial_numbers)
        target = self.trial_data.loc[sel_trial_ind, target_name].tolist()

        if n_time_bins_per_trial is not None:
            target = np.repeat(target, n_time_bins_per_trial)
        else:
            target = np.array(target)

        joinable_targets = ['visualOriPostChangeNorm', 'audioFreqPostChangeNorm']
        if np.isin(target_name, joinable_targets).sum() > 0 and coarse:
            if self.verbose > 1 :
                print('Joining labels (1, 2) and (3, 4)')
            target[np.logical_or(target == 1, target == 2)] = 0
            target[np.logical_or(target == 3, target == 4)] = 1
        if self.verbose > 1:
            print('Returning target {} with {} elements and value '
                  'counts: \n{}'.format(target_name, target.shape[0], pd.value_counts(target)))

        return target

    # --- CELL SELECTION AND METADATA ---
    def select_units(self, area=None, layer=None, min_isolation_distance=None,
                     min_coverage=None, max_perc_isi_spikes=None, celltype=None,
                     return_ids=False):

        # Good units:
        # isolation distance >= 10
        # coverage >= 0.9
        # ISI < 1 %

        total_n_units = self.spike_data['klustaID'].shape[0]
        selected_unit_indices = []

        # if area is not None:
        #     if isinstance(area, str):
        #         area = [area]
        #     mask = np.isin(self.spike_data['area'], area)
        #     sel_trials = self.spike_data.loc[mask, 'trialNum'].tolist()
        #     selected_trials.append(sel_trials)

        if area is not None and area != 'all':
            ind = np.where(self.spike_data['area'] == area)[0]
            selected_unit_indices.append(ind)
            perc = 100*(len(ind) / total_n_units)
            if self.verbose > 1 :
                print('> Selecting units of area {}'
                      '\n ---> {} out of {} units selected '
                      '({:.1f}%)'.format(area, len(ind), total_n_units, perc))

        if layer is not None and layer != 'all_layers':
            assert np.isin(layer, LAYERS)
            try:
                ind = np.where(self.spike_data['layer'] == layer)[0]
            except KeyError:
                ind = np.where(self.spike_data['Layer'] == layer)[0]
            selected_unit_indices.append(ind)
            perc = 100*(len(ind) / total_n_units)
            if self.verbose > 1 :
                print('> Selecting units of layer {}'
                      '\n ---> {} out of {} units selected '
                      '({:.1f}%)'.format(layer, len(ind), total_n_units, perc))


        if min_isolation_distance is not None:
            ind = np.where(self.spike_data['QM_IsolationDistance'] >= min_isolation_distance)[0]
            selected_unit_indices.append(ind)
            perc = 100*(len(ind) / total_n_units)
            if self.verbose > 1 :
                print('> Selecting units with minimum isolation distance of {}'
                      '\n ---> {} out of {} units selected '
                      '({:.1f}%)'.format(min_isolation_distance, len(ind), total_n_units, perc))


        if min_coverage is not None:
            ind = np.where(self.spike_data['coverage'] >= min_coverage)[0]
            selected_unit_indices.append(ind)
            perc = 100*(len(ind) / total_n_units)
            if self.verbose > 1 :
                print('> Selecting units with minimum coverage of {}'
                      '\n ---> {} out of {} units selected '
                      '({:.1f}%)'.format(min_coverage, len(ind), total_n_units, perc))

        if max_perc_isi_spikes is not None:
            perc_isi_spikes = 100 * self.spike_data['QM_ISI_FA']
            ind = np.where(perc_isi_spikes < max_perc_isi_spikes)[0]
            selected_unit_indices.append(ind)
            perc = 100*(len(ind) / total_n_units)
            if self.verbose > 1 :
                print('> Selecting units with less than {}% spikes in the ISI'
                      '\n ---> {} out of {} units selected '
                      '({:.1f}%)'.format(max_perc_isi_spikes, len(ind), total_n_units, perc))


        if celltype is not None:
            ind = np.where(self.spike_data['celltype'] == celltype)[0]
            selected_unit_indices.append(ind)
            perc = 100*(len(ind) / total_n_units)
            if self.verbose > 1 :
                print('> Selecting units with cell type {}'
                      '\n ---> {} out of {} units selected '
                      '({:.1f}%)'.format(celltype, len(ind), total_n_units, perc))


        if len(selected_unit_indices) > 0:
            selected_final = list(set.intersection(*map(set, selected_unit_indices)))
        else:
            selected_final = np.arange(self.spike_data['klustaID'].shape[0]).tolist()

        perc = 100*(len(selected_final) / total_n_units)
        if self.verbose > 1:
            print('> Finished unit selection'
                  '\n ---> {} out of {} units selected '
                  '({:.1f}%)'.format(len(selected_final), total_n_units, perc))

        selected_final.sort()

        if return_ids:
            selected_final = self.get_cell_id(selected_final)

        return selected_final

    def get_layer_from_channel_id(self, channel_id) :
        channel_ind = self.get_lfp_channel_index_from_channel_id(channel_id)
        return self.lfp_data['layer'][channel_ind]

    def get_cell_id(self, cell_index, shortened_id=False) :

        if isinstance(cell_index, int) :
            cell_id = self.cell_id[cell_index]
            if shortened_id :
                cell_id = cell_id[-3 :]
        else :
            cell_id = [self.cell_id[i] for i in cell_index]
            if shortened_id :
                cell_id = [c[-3 :] for c in cell_id]

        return cell_id

    def get_cell_area_from_cell_ind(self, cell_index) :
        return self.spike_data['area'][cell_index]

    def get_cell_area(self, cell_ids) :
        """
        Works for both cell_ids as a single string and as an array of strings
        """
        cell_ind = np.where(np.isin(self.cell_id, cell_ids))[0]
        return self.spike_data['area'][cell_ind]

    def get_cell_layer(self, cell_ids) :
        """
        Works for both cell_ids as a single string and as an array of strings
        """
        cell_ind = np.where(np.isin(self.cell_id, cell_ids))[0]
        return self.spike_data['Layer'][cell_ind]

    def get_cell_depth(self, cell_ids) :
        """
        Works for both cell_ids as a single string and as an array of strings
        """
        cell_ind = np.where(np.isin(self.cell_id, cell_ids))[0]
        return self.spike_data['ChannelY'][cell_ind]

    def get_channel_id_of_cell(self, cell_id):
        cell_ind = np.argwhere(self.cell_id == cell_id)[0]
        channel_id = self.spike_data['channel_ID'][cell_ind]
        return channel_id

    def get_xy_position_of_cell(self, cell_id):
        cell_ind = np.argwhere(self.cell_id == cell_id)[0]
        x = self.lfp_data['ChannelX'][cell_ind]
        y = self.lfp_data['ChannelY'][cell_ind]
        return  np.array([x, y])

    # --- LFP DATA AND METADATA ---


    def select_channels(self, area, layer=None, depth_range=None,
                        in_the_middle=False, q=0.1):

        selected_channel_inds = []
        try:
            ind = np.where(self.lfp_data['area'] == area)[0]
        except KeyError:
            ind = np.where(self.lfp_data['Area'] == area)[0]

        selected_channel_inds.append(ind)

        if layer is not None and not np.isin('all_layers', 'nolfp'):
            assert np.isin(layer, LAYERS)
            try:
                ind = np.where(self.lfp_data['layer'] == layer)[0]
            except KeyError:
                ind = np.where(self.lfp_data['Layer'] == layer)[0]
            selected_channel_inds.append(ind)

        if depth_range is not None:
            mask = np.logical_and(self.lfp_data['ChannelY'] >= depth_range[0],
                                  self.lfp_data['ChannelY'] < depth_range[1])
            ind = np.where(mask)[0]
            selected_channel_inds.append(ind)

        if len(ind) == 0:
            print('There are no channels which match the criteria!')
            return []

        if in_the_middle:
            # the channel depth quantiles are defined only over the channels
            # of the selected area
            channel_depth = self.lfp_data['ChannelY'][ind]
            min_depth = np.quantile(channel_depth, q)
            max_depth = np.quantile(channel_depth, 1-q)
            print(' - Restricting to channels with depth between {:.2f} and '
                  '{:.2f} micrometers'.format(min_depth, max_depth))
            # the indices are across all channels, so we can intersect them
            # with the others
            ind = np.where(np.logical_and(self.lfp_data['ChannelY'] >= min_depth,
                                          self.lfp_data['ChannelY'] <= max_depth))[0]

            selected_channel_inds.append(ind)


        selected_inds = list(set.intersection(*map(set, selected_channel_inds)))

        selected_ids = [self.channel_id[ind] for ind in selected_inds]

        return selected_ids

    def get_lfp_signal_from_channel_index(self, channel_index):
        return self.lfp_data['signal'][channel_index, :]

    def get_lfp_channel_index_from_channel_id(self, channel_ids):
        # This is only for 1 channel
        if isinstance(channel_ids, str):
            channel_ind = np.where(self.channel_id == channel_ids)[0][0]
        else:
            channel_ind = [np.where(self.channel_id == chid)[0][0] for chid in
                           channel_ids]
            channel_ind = np.array(channel_ind)
        return channel_ind

    def get_channel_area(self, channel_ids):
        channel_ind = self.get_lfp_channel_index_from_channel_id(channel_ids)
        return self.lfp_data['area'][channel_ind]

    def get_channel_depth(self, channel_ids):
        channel_ind = self.get_lfp_channel_index_from_channel_id(channel_ids)
        return self.lfp_data['ChannelY'][channel_ind]

    def get_channel_layer(self, channel_ids):
        channel_ind = self.get_lfp_channel_index_from_channel_id(channel_ids)
        return self.lfp_data['Layer'][channel_ind]

    def sort_lfp_channel_ids_by_depth(self, channel_ids):
        channel_ind = self.get_lfp_channel_index_from_channel_id(channel_ids)
        depth = self.lfp_data['ChannelY'][channel_ind]
        sorted_ind = channel_ind[depth.argsort()]
        sorted_channel_ids = self.channel_id[sorted_ind]
        return sorted_channel_ids

    def get_lfp_signal_from_channel_id(self, channel_ids):
        channel_ind = self.get_lfp_channel_index_from_channel_id(channel_ids)
        return self.lfp_data['signal'][channel_ind]

    def get_xy_position_of_channel(self, channel_id):
        channel_ind = np.argwhere(self.channel_id == channel_id)[0]
        x = self.lfp_data['ChannelX'][channel_ind]
        y = self.lfp_data['ChannelY'][channel_ind]
        return np.array([x, y])



    # --- SPIKE BINNING ---

    def get_aligned_times(self, trial_numbers, time_before_in_s=1,
                          time_after_in_s=1, event='stimChange'):

        sel_trial_ind = np.isin(self.trial_data['trialNum'], trial_numbers)
        selected_trial_data = self.trial_data.loc[sel_trial_ind, :]

        time_before_in_us = time_before_in_s * 1e6
        time_after_in_us = time_after_in_s * 1e6

        event_t = selected_trial_data[event].tolist()

        aligned_trial_times = [[s - time_before_in_us, s + time_after_in_us] for
                               s in event_t]

        return aligned_trial_times

    def bin_spikes(self, binsize_in_ms, t_start_in_us=None, t_stop_in_us=None,
                   sliding_window=False, slide_by_in_ms=None):

        """
        Wrapper for the spike binning.

        Relies on two essentially static methods which work fully in
        microseconds. Here the binsize and potentially the sliding window
        are passed in milliseconds which is more intuitive.

        t_start_in_us and t_stop_in_us can be integers OR quantities (we check for it)
        but binsize and slide_by should be integers

        Returns an array of binned spikes and an array of the centers
        of the bins. The bin centers can be used to interpolate phase and
        energy of the lfp.
        """

        # make everything into quantities
        if t_start_in_us is None:
            # session t_start and t_stop are always in microseconds
            t_start_in_us = self.session_t_start
        else:
            try:
                t_start_in_us.rescale(pq.us)
            except AttributeError:
                t_start_in_us = t_start_in_us * pq.us

        if t_stop_in_us is None:
            t_stop_in_us = self.session_t_stop
        else:
            try:
                t_stop_in_us.rescale(pq.us)
            except AttributeError:
                t_stop_in_us = t_stop_in_us * pq.us

        binsize_in_us = binsize_in_ms * 1000 * pq.us

        if slide_by_in_ms is not None:
            slide_by_in_us = slide_by_in_ms * 1000 * pq.us


        # generate spike trains
        spiketrains = self._make_spiketrains(t_start_in_us, t_stop_in_us)

        # bin spikes without sliding window
        if not sliding_window:
            binned_spikes, spike_bin_centers = self._bin_spikes(spiketrains,
                                                                binsize_in_us=binsize_in_us,
                                                                t_start_in_us=t_start_in_us,
                                                                t_stop_in_us=t_stop_in_us)

        # bin spikes with sliding window
        if sliding_window:
            binned_spikes, spike_bin_centers = self._bin_spikes_overlapping(spiketrains,
                                                    binsize_in_us=binsize_in_us,
                                                    slide_by_in_us=slide_by_in_us,
                                                    t_start_in_us=t_start_in_us,
                                                    t_stop_in_us=t_stop_in_us)

        return binned_spikes, spike_bin_centers

    def _make_spiketrains(self, t_start_in_us, t_stop_in_us):
        """
        Input times should have a quantity
        """
        spiketrains = []
        for k in range(self.spike_time_stamps.shape[0]):
            spike_times = self.spike_time_stamps[k]
            spike_times = spike_times[spike_times >= t_start_in_us]
            spike_times = spike_times[spike_times <= t_stop_in_us]
            train = neo.SpikeTrain(times=spike_times * pq.us,
                                   t_start=t_start_in_us,
                                   t_stop=t_stop_in_us)
            spiketrains.append(train)
        return spiketrains

    def _bin_spikes(self, spiketrains, binsize_in_us, t_start_in_us,
                    t_stop_in_us):
        """
        Input times should have a quantity
        """
        bs = elephant.conversion.BinnedSpikeTrain(spiketrains,
                                                  binsize=binsize_in_us,
                                                  t_start=t_start_in_us,
                                                  t_stop=t_stop_in_us)

        n_spikes = np.sum([t.times.__len__() for t in spiketrains])
        n_spikes_binned = bs.to_array().sum()
        if n_spikes != n_spikes_binned:
            warnings.warn('The number of binned spikes is different than '
                          'the number of original spikes')

        binned_spikes = bs.to_array()
        spike_bin_centers = bs.bin_centers.rescale(pq.us)

        return binned_spikes, spike_bin_centers

    def _bin_spikes_overlapping(self, spiketrains, binsize_in_us, slide_by_in_us,
                                t_start_in_us, t_stop_in_us):
        """
        Input times need not be quantities
        """
        try:
            # this needs to be a normal integer for teh list comprehension
            binsize_in_us = binsize_in_us.item()
        except:
            pass

        left_edges = np.arange(t_start_in_us, t_stop_in_us, slide_by_in_us)
        bins = [(e, e + binsize_in_us) for e in left_edges]

        spike_bin_centers = [(b1 + b2) / 2 for b1, b2 in bins]

        # make sure that all the bin centers are within the event
        bins = [b for b in bins if (b[0] + b[1]) / 2 <= (t_stop_in_us)]
        # make sure that bins are fully within
        bins = [b for b in bins if b[1] <= t_stop_in_us]


        # prepare the bin centers (to return)
        spike_bin_centers = [(b1 + b2) / 2 for b1, b2 in bins]

        num_bins = len(bins)  # Number of bins
        num_neurons = len(spiketrains)  # Number of neurons
        binned_spikes = np.empty([num_neurons, num_bins])

        for i, train in enumerate(spiketrains):
            # this is just for safety
            spike_times = train.times.rescale(pq.us)
            for t, bin in enumerate(bins):
                binned_spikes[i, t] = np.histogram(spike_times, bin)[0]

        binned_spikes = binned_spikes.astype(int)
        spike_bin_centers = np.array(spike_bin_centers) * pq.us

        return binned_spikes, spike_bin_centers

    def bin_spikes_per_trial(self, binsize_in_ms, trial_times,
                             sliding_window=False, slide_by_in_ms=None):

        try:
            trial_times.units
        except AttributeError:
            print('Trial times have no units, assuming microseconds (us)')
            trial_times = trial_times * pq.us

        binned_spikes, spike_bin_centers = [], []

        for i, (t_start, t_stop) in enumerate(trial_times):

            bs, bc = self.bin_spikes(binsize_in_ms, t_start_in_us=t_start,
                                     t_stop_in_us=t_stop, sliding_window=sliding_window,
                                     slide_by_in_ms=slide_by_in_ms)
            binned_spikes.append(bs)
            spike_bin_centers.append(bc)

        return binned_spikes, spike_bin_centers



    # --- METADATA OVERVIEWS ---
    def get_percentage_correct(self, trial_type, big_change=False):

        if isinstance(trial_type, str):
            trial_type = [trial_type]

        sel = self.trial_data[np.isin(self.trial_data['trialType'], trial_type)]

        if big_change:
            if trial_type == 'X':
                sel = sel[sel['visualOriChangeNorm'] == 2]
            elif trial_type == 'Y':
                sel = sel[sel['audioFreqChangeNorm'] == 2]

        percentage_correct = 100 * sel['correctResponse'].sum() / sel.shape[0]
        return percentage_correct


    def get_session_overview_trials(self):

        df = pd.DataFrame(columns=['animal_id', 'session_id', 'trial_type',
                                   'target_name', 'only_correct', 'nt',
                                   'nt_0', 'nt_1', 'perc_corr'])

        for trial_type in ['X', 'Y']:
            for only_correct in [True, False]:

                if trial_type == 'X':
                    target_names = ['visualOriPostChangeNorm', 'visualOriChangeNorm']
                if trial_type == 'Y':
                    target_names = ['audioFreqPostChangeNorm', 'audioFreqChangeNorm']

                for target_name in target_names:

                    # this is regardless of
                    perc_corr = self.get_percentage_correct(trial_type=trial_type,
                                                            big_change=True)

                    tn = self.select_trials(only_correct=only_correct,
                                               trial_type=trial_type)

                    y_all = self.make_target(tn, target_name=target_name,
                                                coarse=False)

                    if not np.unique(y_all).shape[0] <= 4:
                        warnings.warn('Session {} {} is FUNKY'.format(self.animal_id,
                                                                      self.session_id))

                    y = self.make_target(tn, target_name=target_name,
                                            coarse=True)
                    nt = len(tn)

                    valcounts = pd.value_counts(y)
                    #print(target_name)
                    #print(valcounts)
                    if valcounts.shape[0] == 2 and np.unique(y_all).shape[0] <= 4:
                        nt_0 = valcounts.iloc[0]
                        nt_1 = valcounts.iloc[1]
                        # assert nt_0 + nt_1 == nt
                        row = [self.animal_id, self.session_id,
                               trial_type, target_name,
                               only_correct, nt, nt_0, nt_1, perc_corr]
                        df.loc[df.shape[0], :] = row

                    else:
                        warnings.warn('Session {} {} for target {} is '
                                      'FUNKY! \nContains values: '.format(self.animal_id,
                                       self.session_id, target_name))
                        print(np.unique(y))


        perc_corr = self.get_percentage_correct(trial_type=['X', 'Y'],
                                                   big_change=True)

        # --- correctResponse target ---
        trial_types = ['X', 'Y']
        only_correct_t = False

        tn = self.select_trials(only_correct=only_correct_t, trial_type=trial_types)
        y = self.make_target(tn, target_name='correctResponse')
        if np.isnan(y).sum() > 0 and self.dataset == 'ChangeDetectionConflictDecor':
            y[np.isnan(y)] = 0
        nt = y.shape[0]
        valcounts = pd.value_counts(y)
        assert valcounts.shape[0] == 2
        nt_0 = valcounts.iloc[0]
        nt_1 = valcounts.iloc[1]
        row = [self.animal_id, self.session_id, None, 'correctResponse',
               only_correct_t, nt, nt_0, nt_1, perc_corr]
        ind = df.shape[0]
        df.loc[ind, :] = row
        df.loc[ind, 'trial_type'] = trial_types

        # --- trialType target ---
        trial_types = ['X', 'Y']
        only_correct_t = False

        tn = self.select_trials(only_correct=only_correct_t, trial_type=trial_types)
        y = self.make_target(tn, target_name='trialType')
        nt = y.shape[0]
        valcounts = pd.value_counts(y)
        assert valcounts.shape[0] == 2
        nt_0 = valcounts.iloc[0]
        nt_1 = valcounts.iloc[1]
        row = [self.animal_id, self.session_id, None, 'trialType',
               only_correct_t, nt, nt_0, nt_1, perc_corr]
        ind = df.shape[0]
        df.loc[ind, :] = row
        df.loc[ind, 'trial_type'] = trial_types

        # --- Lick target ---
        trial_types = ['X', 'Y', 'P']
        only_correct_t = False

        tn = self.select_trials(only_correct=only_correct_t, trial_type=trial_types)
        y = self.make_target(tn, target_name='Lick')
        nt = y.shape[0]
        valcounts = pd.value_counts(y)
        assert valcounts.shape[0] == 2
        nt_0 = valcounts.iloc[0]
        nt_1 = valcounts.iloc[1]
        row = [self.animal_id, self.session_id, None, 'Lick',
               only_correct_t, nt, nt_0, nt_1, perc_corr]
        ind = df.shape[0]
        df.loc[ind, :] = row
        df.loc[ind, 'trial_type'] = trial_types

        # --- Lick Side target ---
        trial_types = ['X', 'Y']
        only_correct_t = True

        try:
            tn = self.select_trials(only_correct=only_correct_t, trial_type=trial_types)
            y = self.make_target(tn, target_name='responseSide')
            nt = y.shape[0]
            valcounts = pd.value_counts(y)
            #print(self.animal_id, self.session_id)
            #assert valcounts.shape[0] == 2
            nt_0 = valcounts.iloc[0]
            nt_1 = valcounts.iloc[1]
            row = [self.animal_id, self.session_id, None, 'responseSide',
                   only_correct_t, nt, nt_0, nt_1, perc_corr]
            ind = df.shape[0]
            df.loc[ind, :] = row
            df.loc[ind, 'trial_type'] = trial_types
        except (NameError, IndexError):
            pass

        # --- Has visual change target ---
        trial_types = ['P', 'X']
        only_correct_t = False

        tn = self.select_trials(only_correct=only_correct_t, trial_type=trial_types)
        y = self.make_target(tn, target_name='hasvisualchange')
        nt = y.shape[0]
        valcounts = pd.value_counts(y)
        assert valcounts.shape[0] == 2
        nt_0 = valcounts.iloc[0]
        nt_1 = valcounts.iloc[1]
        row = [self.animal_id, self.session_id, None, 'hasvisualchange',
               only_correct_t, nt, nt_0, nt_1, perc_corr]
        ind = df.shape[0]
        df.loc[ind, :] = row
        df.loc[ind, 'trial_type'] = trial_types


        # --- Has audio change target ---
        trial_types = ['P', 'Y']
        only_correct_t = False

        tn = self.select_trials(only_correct=only_correct_t, trial_type=trial_types)
        y = self.make_target(tn, target_name='hasaudiochange')
        nt = y.shape[0]
        valcounts = pd.value_counts(y)
        assert valcounts.shape[0] == 2
        nt_0 = valcounts.iloc[0]
        nt_1 = valcounts.iloc[1]
        row = [self.animal_id, self.session_id, None, 'hasaudiochange',
               only_correct_t, nt, nt_0, nt_1, perc_corr]
        ind = df.shape[0]
        df.loc[ind, :] = row
        df.loc[ind, 'trial_type'] = trial_types

        return df


    def get_session_overview_n_units(self):

        df = pd.DataFrame(columns=['animal_id', 'session_id', 'area', 'n_units'])

        for area in ['V1', 'PPC', 'CG1', 'A1']:
            cell_ids = self.select_units(area=area, return_ids=True)
            df.loc[df.shape[0], :] = [self.animal_id, self.session_id, area,
                                      len(cell_ids)]
        return df


    def get_session_overview_units_and_channels(self):

        df = pd.DataFrame(columns=['animal_id', 'session_id', 'area', 'layer',
                                   'n_units', 'n_channels'])
        for area in ['V1', 'PPC', 'CG1', 'A1']:
            for layer in ['IG', 'SG', 'NA']:
                chan_ids = self.select_channels(area=area, layer=layer)
                cell_ids = self.select_units(area=area, layer=layer, return_ids=True)
                df.loc[df.shape[0], :] = [self.animal_id, self.session_id, area, layer,
                                          len(cell_ids), len(chan_ids)]

        return df



def select_sessions(dataset, area, min_units, min_performance=None,
                    filter_performance=False,
                    exclude_muscimol=False, exclude_opto=False):

    """
    basic utility function to select multiple sessions based on criteria
    """

    sdf = pd.DataFrame(columns=['animal_id', 'session_id', 'layer', 'n_units',
                                'MuscimolArea', 'UseOpto', 'PhotostimArea', 'perf_audio',
                                'perf_visual'])

    for animal_id in all_sessions[dataset].keys() :
        for session_id in all_sessions[dataset][animal_id] :
            try :
                session = Session(dataset=dataset, animal_id=animal_id,
                                  session_id=session_id, subfolder='')
                session.load_data(load_lfp=False, load_lfp_lazy=False,
                                  load_spikes=True)
            except OSError :
                print('skipping {} {}'.format(animal_id, session_id))
                continue

            units = session.get_session_overview_n_units()
            n_units = units[units['area'] == area]['n_units'].iloc[0]
            trialoverview = session.get_session_overview_trials()
            perf_audio = trialoverview[trialoverview['trial_type'] == 'X'].iloc[0]['perc_corr']
            perf_visual = trialoverview[trialoverview['trial_type'] == 'Y'].iloc[0]['perc_corr']

            df = session.trial_data
            muscimol = session.session_data['MuscimolArea']
            useopto = session.session_data['UseOpto']
            photostimarea = session.session_data['PhotostimArea']

            if np.isnan(session.trial_data['visualOriPostChange']).sum() > 0:
                warnings.warn('Session {} {} has no visual info - SKIPPING)'.format(animal_id, session_id))
                continue


            sdf.loc[sdf.shape[0], :] = [animal_id, session_id, None, n_units,
                                        muscimol, useopto, photostimarea, perf_audio, perf_visual]


    sdf['area'] = area

    sdf = sdf[(sdf['n_units'] >= min_units)]

    if exclude_muscimol:
        # select no muscimol
        sdf = sdf[sdf['MuscimolArea'].isna()]

    if exclude_opto:
        # select no opto in the selected area
        sdf = sdf[sdf['PhotostimArea'] != area]

    if filter_performance and min_performance is not None:
        sdf = sdf[(sdf['perf_audio'] >= min_performance) &
                  (sdf['perf_visual'] >= min_performance)]

    return sdf

# %%

if __name__ == '__main__':

    animal_id = '2012'
    session_id = '2018-08-13_14-31-39'
    dataset = 'ChangeDetectionConflict'

    session = Session(dataset=dataset, animal_id=animal_id,
                      session_id=session_id, subfolder='')

    session.load_data(load_lfp=False, use_newtrials=True)

    # get an overview of the cells recorded in the session
    print(session.get_session_overview_n_units())

    # get an overview of the number of trials per type
    # for example, the first row indicates that there are 27 correct visual trials,
    # and if you split them based on the post-change visual orientation
    # (nearby orientation grouped together) you have 18 and 9 trials of either type
    session.get_session_overview_trials()

    # get an overview of the cells and corresponding recording channels
    # session.get_session_overview_units_and_channels()

    # this is where the metadata about individual trials is
    session.trial_data

    # and about the recording session
    session.session_data

    # and this is where the spike data is

    # a typical workflow include selecting trials
    trial_numbers = session.select_trials(only_correct=False,
                                          trial_type='Y', # audio
                                          auditory_change=3) # large auditory change

    # getting beginning and end times for every trial aligned to stimulus change
    trial_times = session.get_aligned_times(trial_numbers,
                                            time_before_in_s=1, # 1 sec before stim. change
                                            time_after_in_s=2, # to 2 sec after stim. change
                                            event='stimChange')

    # binning spikes (e.g. with a 200 ms window advanced in 100ms time increments)
    binned_spikes, spike_bin_centers = session.bin_spikes_per_trial(binsize_in_ms=200,
                                                                    trial_times=trial_times,
                                                                    sliding_window=True,
                                                                    slide_by_in_ms=100)
    # binned_spikes is a list, every element is an array of shape n_units x n_time_points


    # --- CV ---
    # Parameters
    area = 'PPC'
    trial_type = 'X'
    size_of_change = 3
    min_nr_spikes = 3
    sec_pre_post = 2


    # Selecting Units
    selected_unit_ind = session.select_units(area=area, layer=None)

    # Selecting Trials

    if trial_type == 'X':
        trial_numbers = session.select_trials(only_correct=False,
                                            trial_type='X', # visual
                                            visual_change=3) # visual change
    if trial_type == 'Y':
        trial_numbers = session.select_trials(only_correct=False,
                                            trial_type='Y', # auditory
                                            auditory_change=3) # auditory change

    trial_times = session.get_aligned_times(trial_numbers,
                                            time_before_in_s=sec_pre_post, # sec before stim. change
                                            time_after_in_s=sec_pre_post, # to sec after stim. change
                                            event='stimChange')
    # Spikes
    all_spikes = session.spike_time_stamps

    # Initialize Result Matrices
    mu_results = np.ones((len(selected_unit_ind),len(trial_times),2)) * sec_pre_post * 1000/min_nr_spikes # Filled with maximal measurable ITI
    std_results = np.zeros((len(selected_unit_ind),len(trial_times),2))
    # Neurons
    for nn,n in enumerate(selected_unit_ind):
        unit_spikes = all_spikes[nn]

        # Trials
        for tn, (trial_start, trial_stop) in enumerate(trial_times):

            # Trial
            trial_spikes = unit_spikes[unit_spikes>=trial_start]
            trial_spikes = trial_spikes[trial_spikes<=trial_stop]
            trial_length = trial_stop - trial_start
            trial_mid = trial_start + trial_length/2

            # Before
            before_stim = trial_spikes[trial_spikes<=trial_mid]
            if len(before_stim)>min_nr_spikes:
                mu_iti_before = np.mean(np.diff(before_stim)/1000)
                std_iti_before = np.std(np.diff(before_stim)/1000)
                mu_results[nn,tn,0] = mu_iti_before
                std_results[nn,tn,0] = std_iti_before

            # After
            after_stim = trial_spikes[trial_spikes>=trial_mid]
            if len(after_stim)>min_nr_spikes:
                mu_iti_after = np.mean(np.diff(after_stim)/1000)
                std_iti_after = np.std(np.diff(after_stim)/1000)
                mu_results[nn,tn,1] = mu_iti_after
                std_results[nn,tn,1] = std_iti_after

    # Per neuron
    neuron = 5
    blub = mu_results[neuron,:,0]
    blub2 = mu_results[neuron,:,1]

    remove_till = 25
    mu_results = mu_results[:,:remove_till]
    std_results = std_results[:,:remove_till]

    plt.subplot(211)
    plt.plot(mu_results[neuron].T,marker='o')

    plt.subplot(212)
    plt.plot(mu_results[neuron],marker='o')

    plt.show()

    mu_across_trials = np.mean(mu_results,axis=1)
    std_across_trials = np.mean(std_results,axis=1)

    print(mu_across_trials)
    print(mu_across_trials)
    print(std_across_trials)
    print(std_across_trials)

    mu_across_neurons = np.mean(mu_across_trials,axis=0)
    std_across_neurons = np.mean(std_across_trials,axis=0)

    print(mu_across_trials)
    print(std_across_trials)
    print(mu_across_neurons)
    print(std_across_neurons)

    fig, ax = plt.subplots()
    ax.scatter(std_across_trials[:,0],mu_across_trials[:,0],color='red')
    ax.scatter(std_across_trials[:,1],mu_across_trials[:,1],color='blue')
    plt.show()

    # Finding Neuron parameters
    neuron_params = {'C_m': 1.0,
                    't_ref': 0.1,
                    'V_reset': -65.0,
                    'tau_m': 10.0,
                    'V_th': -50.0,
                    'E_L': -65.0}

    # Simulation Parameters
    sim_params = {'dt_noise': 0.01,
                'sim_res': 0.01,
                'mean_mem': -65.0,
                'std_mem': 20,
                'simtime': 50000,
                'seed': 18,
                'neurons': 50,
                'search_start': [-75, 23]}





    # --- FANO factor ---
    # set time parameters
    binsize_in_ms = 500
    time_before_stim_in_s = 1
    time_after_stim_in_s = 2
    slide_by_in_ms = 10
    sliding_window = True

    # select trials of a given type
    trial_numbers = session.select_trials(only_correct=False,
                                          trial_type='X', # visual
                                          visual_change=3) # large visual change

    # getting beginning and end times for every trial aligned to stimulus change
    trial_times = session.get_aligned_times(trial_numbers,
                                            time_before_in_s=time_before_stim_in_s, # 1 sec before stim. change
                                            time_after_in_s=time_after_stim_in_s, # to 2 sec after stim. change
                                            event='stimChange')

    # require a certain minimum amount of trials
    assert len(trial_times) > 10

    # binning spikes (e.g. with a 100 ms window advanced in 10ms time increments)
    binned_spikes, spike_bin_centers = session.bin_spikes_per_trial(binsize_in_ms=binsize_in_ms,
                                                                    trial_times=trial_times,
                                                                    sliding_window=sliding_window,
                                                                    slide_by_in_ms=slide_by_in_ms)

    # get the time of each bin relative to stimulus onset
    time_bin_times = (spike_bin_centers[0] - spike_bin_centers[0][0]).rescale(pq.s) + (binsize_in_ms * pq.ms).rescale(pq.s) / 2 - time_before_stim_in_s * pq.s
    time_bin_times = time_bin_times.__array__().round(5)

    # subselecting units from a given area with required quality metrics
    # see select_units for other available selection criteria (e.g. coverage)
    # and see session.spike_data.keys() for all the available unit metrics
    selected_unit_ind = session.select_units(area='PPC', layer=None,
                                             min_isolation_distance=10,
                                             max_perc_isi_spikes=0.1)

    # # get the unique unit ID
    # selected_unit_id = session.get_cell_id(selected_unit_ind, shortened_id=False)

    # # selecting the binned spike only of the selected units
    # binned_spikes = [s[selected_unit_ind, :] for s in binned_spikes]

    # # make an empty array
    # fano_arr = np.zeros(shape=(len(selected_unit_id), len(time_bin_times)))

    # # compute the fano factor for every unit and every time point
    # for u, id in enumerate(selected_unit_id):
    #     for t, time in enumerate(time_bin_times):
    #         spikes = np.array([s[u, t] for s in binned_spikes])
    #         fano_arr[u, t] = spikes.var() / spikes.mean()

    # # average fano factor across units
    # fano = fano_arr.mean(axis=0)

    # # plot the results
    # f, ax = plt.subplots(1, 1, figsize=[3, 3])
    # ax.plot(time_bin_times, fano)
    # ax.axvline(0, c='grey', ls='--')
    # sns.despine()
    # ax.set_ylabel('Fano factor')
    # ax.set_xlabel('Time [s]')
    # plt.tight_layout()


    # # --- SELECTING SESSIONS ---
    # # we can check whether in this session there was muscimol or optogenetic
    # # stimulation by checking the fields
    # session.session_data['MuscimolArea'] # if nan, no muscimol
    # session.session_data['UseOpto'] # if 0, no opto
    # session.session_data['PhotostimArea'] # area of opto stimulation

    # # we can also use select_sessions to get all sessions with at least
    # # one PPC unit
    # sessions = select_sessions(dataset='ChangeDetectionConflict',
    #                 area='PPC', min_units=1)

    # # or already exclude sessions which have muscimol or opto in PPC
    # sessions_nomus_noopto = select_sessions(dataset='ChangeDetectionConflict',
    #                         area='PPC', min_units=1, exclude_muscimol=True,
    #                             exclude_opto=True)

    # plt.show()
# %%
