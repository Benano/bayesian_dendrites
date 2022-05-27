import os
import platform


# --- PATHS --------------------------------------------------------------------
# figure out the paths depending on which machine you are on

user = os.environ['USER']
print(user)

if user == 'pietro':

    if platform.system() == 'Linux':
        jarLocation = '/home/pietro/code/jidt/infodynamics.jar'
        DATA_FOLDER = '/media/pietro/bigdata/neuraldata/modid'

    elif platform.system() == 'Darwin':
        #DATA_FOLDER = '/Users/pietro/data/modid'
        DATA_FOLDER = '/Users/pietro/surfdrive/Shared/Data/CHDET'

elif user == 'pmarche1':
    jarLocation = '/home/pmarche1/code/jidt/infodynamics.jar'

elif user == 'benano':
    DATA_FOLDER = "/Users/Benano/data/pennartz_collab"


# -----------------------------------------------------------------------------

AREAS = ['V1', 'PPC', 'CG1']
LAYERS = ['SG', 'G', 'IG', 'NA']

# --- SESSIONS ----------------------------------------------------------------

# TODO these may be incomplete

# list all decode sessions

changed_audio_presentation_animals = ['2019',
                                      '2020',
                                      '2021',
                                      '2028',
                                      '2029',
                                      '2034',
                                      '2035']

doubleshank_sessions = ['2018-08-23_13-20-25',
                        '2018-08-14_12-24-28',
                        '2018-08-15_10-09-24',
                        '2018-08-14_14-30-15',
                        '2018-08-15_14-13-50',
                        '2018-08-23_16-18-45']

sessions_change_detection_conflict_freq = {
    '2003' : ['2018-02-07_14-28-57',
           '2018-02-08_15-13-56',
           '2018-02-09_14-30-19'],
    # '2004' : ['2018-02-15_14-57-44',  # no spikedata
    #          '2018-02-16_15-29-18'], # no spikedata
    '2009' : ['2018-08-22_14-05-50',
           '2018-08-23_13-20-25',
            '2018-08-24_11-56-35'], #no visual orientations
    '2010' : ['2018-08-13_11-21-09',
           '2018-08-14_12-24-28',
           '2018-08-15_10-09-24', #no lfp!
           '2018-08-16_09-52-11'],
    '2011' : ['2018-08-08_10-12-14',
           '2018-08-09_11-01-15',
           '2018-08-10_10-14-01'],
    '2012' : ['2018-08-13_14-31-39',
           '2018-08-14_14-30-15',
           '2018-08-15_14-13-50',
           '2018-08-16_11-54-39'],
    '2013' : ['2018-08-23_16-18-45',
           '2018-08-24_13-24-37']} #no visual orientations

sessions_change_detection_conflict_octaves = {
    '2019' : ['2019-06-26_09-31-34',
              '2019-06-27_10-33-24'],
    '2020' : ['2019-07-03_12-16-46',
              '2019-07-01_10-48-52',
              '2019-06-30_13-40-02',
              '2019-07-04_09-13-46'],
    '2023' : ['2019-07-03_17-31-18',
              '2019-07-02_14-55-22',
              '2019-07-01_14-29-07'],
    '2021' : ['2019-06-30_15-36-18',
              '2019-07-01_12-51-10',
              '2019-07-03_15-06-59',
              '2019-07-04_11-16-25'],
    '2022' : ['2019-06-26_11-57-02',
              '2019-06-27_13-52-56',
              '2019-06-28_12-27-33',
              '2019-07-01_16-55-03',
              '2019-07-16_14-32-06',
              '2019-07-17_14-06-30'],
    '2023' : ['2019-07-01_14-29-07',
              '2019-07-02_14-55-22',
              '2019-07-03_17-31-18'],
    '2030' : ['2020-01-22_11-49-17'],
    '2031' : ['2020-01-23_16-24-17',
              '2020-01-24_15-19-14']}

sessions_change_detection_conflict_octaves_NEW = {
    '2044' : ['2021-04-20_16-48-03',
              '2021-04-22_13-45-10',
              '2021-04-23_13-12-02',
              #'2021-04-24_14-48-22', # TODO no session data
              #'2021-04-24_14-48-24', # TODO was breaking with train test pseudodec
              #'2021-04-28_14-00-10', # TODO no session data
              #'2021-04-28_14-00-12'  # TODO was breaking with train test pseudodec -> only incorrect response
              ],
    '2005' : ['2021-04-20_18-27-54',
              '2021-04-22_15-49-51',
              '2021-04-23_17-48-52',
              '2021-04-28_19-13-59',
              '2021-04-28_19-13-61',
              '2021-04-29_15-55-10',
              '2021-04-29_15-55-12',
              '2021-04-30_15-26-10',
              '2021-04-30_15-26-12']
}


sessions_change_detection_conflict_decor = {
    '1008' : ['2019-03-07_10-23-36', #no visual orientations
              '2019-03-08_12-38-20',
              '2019-03-12_11-28-33',
              '2019-03-13_13-01-19'],
    '1009' : ['2019-03-06_14-12-13',  #no visual orientations
              '2019-03-07_12-46-53', #no visual orientations
              '2019-03-08_14-39-42',
              '2019-03-12_15-41-39',
              '2019-03-13_15-32-33'],
    '1012' : ['2019-04-09_11-23-04',
              '2019-04-10_11-43-25',
              '2019-04-11_11-12-47',
              '2019-04-12_09-04-51'],
}

sessions_change_detection_volume = {
    '2020' : ['2019-07-01_10-48-52'],
    '2021' : ['2019-07-02_13-01-51'],
    #'2022' : [],
}


sessions_visonlytwolevels = {'2028' : ['2019-12-11_16-08-25',
                                       '2019-12-12_15-54-10',
                                       '2019-12-13_12-46-26',
                                       '2019-12-16_11-12-25'],
                             '2029' : ['2019-12-11_13-49-32',
                                       '2019-12-12_13-45-39',
                                       '2019-12-13_09-59-24'],
                             '2034' : ['2019-12-18_13-31-12',
                                       '2019-12-19_13-50-12'],
                             '2035' : ['2019-12-16_15-38-43',
                                       '2019-12-18_15-44-47',
                                       '2019-12-19_15-28-39']}

sessions_change_detection_conflict = {**sessions_change_detection_conflict_freq ,
                                      **sessions_change_detection_conflict_octaves,
                                      **sessions_change_detection_conflict_octaves_NEW}


all_sessions = {'ChangeDetectionConflict' : sessions_change_detection_conflict,
                'ChangeDetectionConflictFreq' : sessions_change_detection_conflict_freq,
                'ChangeDetectionConflictOctaves' : sessions_change_detection_conflict_octaves,
                'ChangeDetectionConflictDecor' : sessions_change_detection_conflict_decor,
                'ChangeDetectionVolume' : sessions_change_detection_volume,
                'VisOnlyTwolevels' : sessions_visonlytwolevels}


