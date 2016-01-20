__author__ = 'bramzandbelt'

# Import packages
from psychopy import visual, monitors, core, event, iohub, info
print 'psychopy version: %s' % info.psychopyVersion
import os
import time
import calendar
from itertools import chain

import numpy as np
print 'numpy version: %s' % np.__version__
import pandas as pd
print 'pandas version: %s' % pd.__version__
from pandas import DataFrame
# This is to enable printing of all data frame columns
pd.set_option('display.max_columns', None)

from pprint import pprint
import re


# To Do:
#
# ADD PAUSE KEY FUNCTIONALITY
# =============================================================================
# if pauseKey.keys:
#   td = win._toDraw
#   win._toDraw = []  # hides whatever was being auto-drawn
#   txt = visual.TextStim(win,text='(paused, p to continue)')
#   while not event.getKeys(keyList=['p']):
#     txt.draw()
#     win.flip()
#   win._toDraw = td  # restore auto-draw
#   pauseKey.status = NOT_STARTED
#   pauseKey.keys = []

# Some general stuff
# =============================================================================

# Do not monitor for key presses during wait periods
core.checkPygletDuringWait = False

def collect_response(rd,kb, *args, **kwargs):
    """
    Collect responses from response device and keyboard

    This function is called at two stages in the experiment:
    1. Instruction
        Returns count of response keys, escape keys, and toggle keys
    2. Experiment
        Updates the log variable with key count and key time information

    Parameters
    ----------
    rd : type
        blablabla
    kb : type
        blablabla

    Returns
    -------
    log : type
        blablabla

    log = collect_response(rd,kb,log)

    keycount : type
        blablabla

    keyCount = collect_response(rd,kb)

    Raises
    ------

    USAGE
    # For collecting experimental data
    log = collect_response(rd,kb,log)

    # For instruction screens
    keyCount = collect_response(rd,kb)

    """

    # TODO: Determine why responses from Serial response box result in incorrect response, even though the right buttons were pressed

    if len(args) == 0:
        if len(kwargs) == 0:
            toggleKeyPressed = None
        elif len(kwargs) == 1:
            log = kwargs.get('log')
        elif len(kwargs) > 1:
            # TODO: Add error message
            pass
    elif len(args) == 1:
        log = args[0]
    elif len(args) > 1:
        # TODO: Add error message
        pass

    # Process inputs
    # -------------------------------------------------------------------------
    rspKeys     = [item
                   for sublist in rd['settings']['rspKeys']
                   for item in sublist]
    keyKey      = rd['settings']['keyKey']
    timeKey     = rd['settings']['timeKey']
    rdClass     = rd['settings']['class']

    escKeys     = kb['settings']['escKeys']
    toggleKeys  = kb['settings']['toggleKeys']

    # Define dynamic variables
    # -------------------------------------------------------------------------
    keyCount    = {key: 0 for key in rspKeys}
    keyTime     = {key: [] for key in rspKeys}

    # Determine response identity and response
    # -------------------------------------------------------------------------
    rdEvents = rd['client'].getEvents()

    for ev in rdEvents:

        evKeys = ev._asdict()[keyKey]
        evTime = ev._asdict()[timeKey]

        # If response device is keyboard and escape keys are in event data
        if rdClass == 'Keyboard':
            if any([len(re.findall(key,ev._asdict()['key'])) for key in escKeys]):
                print('Warning: Escape key pressed - Experiment is terminated')
                core.quit()
            if any([len(re.findall(key,ev._asdict()['key'])) for key in toggleKeys]):
                toggleKeyPressed = ev._asdict()['key']
                return keyCount, toggleKeyPressed

        # If any of the event keys are in event data
        if any([len(re.findall(key,evKeys)) for key in rspKeys]):
            for key in rspKeys:
                keyCount[key] += evKeys.count(key)
                if len(args) == 1 or len(kwargs) == 1:
                    keyTime[key].extend([evTime - log.iloc[0]['trialOns']] * evKeys.count(key))

    # Check if any escape or toggle keys have been pressed
    # -------------------------------------------------------------------------
    if rdClass != 'Keyboard':

        for ev in kb['client'].getEvents():
            if any([len(re.findall(key,ev._asdict()['key'])) for key in escKeys]):
                print('Warning: Escape key pressed - Experiment is terminated')
                core.quit()
            if any([len(re.findall(key,ev._asdict()['key'])) for key in toggleKeys]):
                toggleKeyPressed = ev._asdict()['key']
                return keyCount, toggleKeyPressed

    # For each key, only response times of first two events are stored
    if len(args) == 1 or len(kwargs) == 1:

        for key in rspKeys:
            if keyCount[key] > 2:
                keyTime[key] = keyTime[key][0:2]
            elif keyCount[key] < 2:
                keyTime[key].extend([float('NaN')] * (2 - keyCount[key]))

            # Convert to numpy ndarray
            keyTime[key] = np.array(keyTime[key])

        # Log response events
        for key in rspKeys:
            log.iloc[0]['keyCount_'+key] = keyCount[key]
            log.iloc[0]['keyTime1_'+key] = keyTime[key][0]
            log.iloc[0]['keyTime2_'+key] = keyTime[key][1]

        return log

    else:

        return keyCount, toggleKeyPressed

def compute_trial_statistics(trialStats,rd,log):
    """
    Computes descriptive statistics, such as response time, response time
    difference, and raw processing time

    Parameters
    ----------


    Returns
    -------


    Raises
    ------


    """

    rspKeys = [item for sublist in rd['settings']['rspKeys'] for item in sublist]
    rspKeyPairs = rd['settings']['rspKeys']

    s1Ons = log.iloc[0]['s1Ons']
    s2Ons = log.iloc[0]['s2Ons']

    # Response time
    # -------------------------------------------------------------------------
    rt1 = {key: np.nan for key in rspKeys}
    rt2 = rt1.copy()

    for key in rspKeys:
        rt1[key] = log.iloc[0]['keyTime1_'+key] - s1Ons
        rt2[key] = log.iloc[0]['keyTime2_'+key] - s1Ons

        if trialStats['rt']:
            log.iloc[0]['rt1_'+key] = rt1[key]
            log.iloc[0]['rt2_'+key] = rt2[key]

    # Response time difference
    # -------------------------------------------------------------------------
    if trialStats['rtDiff']:
        for pair in rspKeyPairs:
            pairStr = pair[0] + '-' + pair[1]
            log.iloc[0]['rtDiff1_' + pairStr] = rt1[pair[0]] - rt1[pair[1]]
            log.iloc[0]['rtDiff2_' + pairStr] = rt2[pair[0]] - rt2[pair[1]]

    # Raw processing time
    # -------------------------------------------------------------------------
    if trialStats['rpt']:
        for key in rspKeys:
            log.iloc[0]['rpt1_' + key] = rt1[key] - s2Ons
            log.iloc[0]['rpt2_' + key] = rt2[key] - s2Ons

    return log

def evaluate_trial(evalData,window,stimuli,log):
    """

    Parameters
    ----------
    evalData : type
        blablabla
    window : type
        blablabla
    stimuli : type
        blablabla
    log : type
        blablabla

    Returns
    -------

    Raises
    ------

    """

    # Process inputs
    # =========================================================================

    # Assertions
    # -------------------------------------------------------------------------

    # specific to evalData and log (window and stimulis should have been
    # checked already and not have been updated

    # Define dynamic variables
    # -------------------------------------------------------------------------
    trialLabel      = []
    trialFeedback   = []

    # Trial evaluation
    # =========================================================================
    source = evalData['evalData'].fillna(float('inf'))
    patternDict = log[source.columns].fillna(float('inf')).to_dict()
    ix = log.index.tolist()[0]
    pattern = {key: [value[ix]] for key,value in patternDict.iteritems()}

    iRow = source.isin(pattern).all(1)

    labelValue = evalData['label'].loc[iRow].as_matrix().tolist()

    if sum(iRow) == 0:
        trialLabel = 'no match'
        trialFeedback = 'incorrect response (no match)'

        print 'patternDict for incorrect response (no match): \n'
        pprint(patternDict)

    if sum(iRow) == 1:
        trialLabel = ", ".join(labelValue)
        trialFeedback = evalData['feedback'].loc[iRow].as_matrix().tolist()[0]

    if sum(iRow) > 1:
        trialLabel = 'multi match: ' + ", ".join(labelValue)
        trialFeedback = 'incorrect response (multi match)'
        print 'WARNING: Non-unique trial label'

    log.iloc[0]['trialType'] = trialLabel
    log.iloc[0]['trialFeedback'] = trialFeedback

    # SOA adjustments
    # =========================================================================

    # newSoa = []
    #
    # if soa['type'] == 'staircase':
    #
    #     soaIx = []
    #     soa = []
    #
    #
    #     if trialLabel == 'correct':
    #         newSoa = soa + soaStep
    #     else:
    #         newSoa = soa - soaStep

    # Feedback
    # =========================================================================

    stimuli['feedback'][0].setText(trialFeedback)
    stimuli['feedback'][0].setAutoDraw(True)
    stimuli['iti'][0].setAutoDraw(True)
    window.flip()
    core.wait(0.5)
    stimuli['feedback'][0].setAutoDraw(False)
    stimuli['iti'][0].setAutoDraw(True)
    window.flip()
    stimuli['iti'][0].setAutoDraw(False)

    return log

def get_empty_text_stim(window):

    """

    Parameters
    ----------
    window

    INPUTS
    - window
    - text
    - font

    THOUGHTS:
    - use key-value pairs?

    :return:
    """

    textStim = visual.TextStim(window,
                        text='',
                        font='Arial',
                        pos=(0,0),
                        color=[1.0, 1.0, 1.0],
                        colorSpace='rgb',
                        opacity=1.0,
                        bold=False,
                        italic=False,
                        alignHoriz='center',
                        alignVert='center',
                        wrapWidth=None,
                        autoLog=None)

    textStim.setSize(2,units='deg')

    return textStim

def define_stimulus(window,stimInfo):
    """
    Make PsychoPy stimuli

    Currently, only stimuli of type TextStim and ImageStim are supported.

    Parameters
    ----------
    window : class?
        window where stimuli will be shown
    stimInfo : dict
        key-value pairs

    Returns
    -------
    stimulus:

    Raises
    ------

    """

    # Process inputs
    # -------------------------------------------------------------------------

    # TODO: See if assertions require __debug__ variable
    assert type(window) is visual.window.Window, \
        'window is not of class visual.window.Window'
    assert type(stimInfo) is dict, \
        'stimInfo is not of type dict'
    for key in ['type','content','name']:
        assert stimInfo.has_key(key), \
            'stimInfo does not contain key {0:s}'.format(key)

    nStimulus = len(stimInfo['content'])

    # Initialize list of stimuli
    stimulus = [None] * nStimulus

    for i in range(nStimulus):

        stimulus[i] = init_stimulus(window,stimInfo['type'])

        assert (type(stimulus[i]) is
                visual.text.TextStim or
                visual.image.ImageStim), \
                "stimulus is neither a TextStim nor an ImageStim"

        # Set stimulus name
        stimulus[i].name = stimInfo['name']

        # Text stimuli
        # ---------------------------------------------------------------------
        if type(stimulus[i]) is visual.text.TextStim:

            # Set stimulus content
            if os.path.isfile(stimInfo['content'][i]):
                with open(stimInfo['content'][i]) as txtFile:
                    stimulus[i].setText(txtFile.read())
            else:
                stimulus[i].setText(stimInfo['content'][i])

            # Set other parameters
            if 'fontFile' in stimInfo:
                if stimInfo['fontFile']:
                    stimulus[i].fontFiles = stimInfo['fontFile']
            if 'font' in stimInfo:
                if stimInfo['font']:
                    stimulus[i].setFont(stimInfo['font'])
            if 'ori' in stimInfo:
                if stimInfo['ori']:
                    stimulus[i].setOri(stimInfo['ori'])
            if 'height' in stimInfo:
                if stimInfo['height']:
                    stimulus[i].setHeight(stimInfo['height'])
            if 'pos' in stimInfo:
                if stimInfo['pos']:
                    stimulus[i].setPos(stimInfo['pos'])
            if 'color' in stimInfo:
                if stimInfo['color']:

                    # Check if color vary or be the same across different stimuli of this type
                    if all([isinstance(stimInfo['color'],list),len(stimInfo['color']) == 3,isinstance(stimInfo['color'][0],int)]):
                        stimulus[i].setColor(stimInfo['color'], 'rgb255')
                    elif all([isinstance(stimInfo['color'],list),len(stimInfo['color']) == len(stimInfo['content']),isinstance(stimInfo['color'][0],list)]):
                        stimulus[i].setColor(stimInfo['color'][i], 'rgb255')

        # Image stimuli
        # ---------------------------------------------------------------------
        elif type(stimulus[i]) is visual.image.ImageStim:

            # Set stimulus content
            stimulus[i].setImage(stimInfo['content'][i])

            # Set other parameters
            if stimInfo['ori']:
                stimulus[i].setOri(stimInfo['ori'])
            if stimInfo['pos']:
                stimulus[i].setPos(stimInfo['pos'])

    return stimulus

def stim_to_frame_mat(config,trial,log):
    """

    :param config:
    :param trial:
    :param log:
    :return t:
    :return u:

    """

    trialLogIx = log.index.tolist()[0]

    stimuli = config['stimuli']

    def append_it(trial,log,stim,stimList,ons,dur):
        """

        :param trial:
        :param stim:
        :param stimList:
        :param ons:
        :param dur:
        :param log:

        :return stimList:
        :return ons:
        :return dur:
        :return log:
        :return iStim:
        """

        if not pd.isnull(trial[stim + 'Ix']):

            # Identify stimulus index
            i = trial.loc[stim + 'Ix'].astype(int)

            # Append stimulus list, onset array, and duration array
            stimList.append(stimuli[stim][i])
            stimOns    = float(trial[stim + 'Ons'])
            stimDur    = float(trial[stim + 'Dur'])
            newOns     = np.hstack([ons,stimOns])
            newDur     = np.hstack([dur,stimDur])

            log.loc[trialLogIx,[stim + 'Ix']] = i

            # N.B. These are the intended stimulus onsets and durations. They
            # are replace by the actual onsets and durations in present_stimuli
            log.loc[trialLogIx,[stim + 'Ons']] = stimOns
            log.loc[trialLogIx,[stim + 'Dur']] = stimDur

            if stim == 's2':
                log.loc[trialLogIx,['soaIx']] = trial.loc['soaIx'].astype(int)

        else:
            newOns = ons
            newDur = dur

            for col in ['Ix','Ons','OnsDt','Dur','DurDt']:
                log.loc[trialLogIx,[stim + col]] = np.nan

            if stim == 's2':
                log.loc[trialLogIx,['soaIx']]   = np.nan


        return stimList, newOns, newDur, log

    stimList    = []
    ons         = np.array([],dtype=int)
    dur         = np.array([],dtype=int)

    stimSet     = set(config['stimuli'].keys())

    for stim in stimSet.intersection(['fix','cue','s1','s2']):
        stimList, ons, dur, log = append_it(trial=trial,
                                            log=log,
                                            stim=stim,
                                            stimList=stimList,
                                            ons=ons,
                                            dur=dur)

    # Append iti stimulus to stimList
    stimList.append(stimuli['iti'][0])

    dt = np.array([1./config['window']['frameRate']])
    t_max = np.array(np.max(ons + dur))

    # Make stimulus-by-frame matrix (u)
    u, f_on_off, t = time_to_frame(ons=ons, dur=dur, dt=dt, t_max=t_max)

    return log, stimList, u, f_on_off, t_max


def get_rt_diff(keyRt,pair):
    """

    :param keyRt: dictionary with response times per key
    :param pair: list of response key pairs
    :return rtDiff: difference between first response times
    """

    rt = [keyRt[key][0] for key in pair]

    if all(rt):
        rtDiff = rt[0] - rt[1]
    else:
        rtDiff = np.nan

    return rtDiff

def get_trial_list(argument):
    """

    :return:
    """

    switcher = {'load': load_trial_list(), 'make': make_trial_list()}

    # Get the function from switcher dictionary
    func = switcher.get(argument)
    # TODO: enter exception, when argument is not found in dictionary

    # Execute the function
    return func()

def init_log(config):
    """
    Initializes a pandas DataFrame for logging stop task data

    Parameters
    ----------
    config : dict
        configuration dictionary, containing experiment information

    Returns
    -------
    log : pandas.core.frame.DataFrame

    """

    # Process inputs
    # -------------------------------------------------------------------------


    # Assertions



    index   = []
    columns = []

    # Define session columns and data
    # =========================================================================
    idColumnsSess   = ['studyId',
                       'experimenterId',
                       'responseDevice']
    idDataSess      = [config['study']['studyId'],
                       config['session']['experimenterId'],
                       config['apparatus']['rd']['settings']['class']]

    ixColumnsSess   = ['subjectIx',
                       'groupIx',
                       'sessionIx']
    ixDataSess      = [config['subject']['subjectIx'],
                       config['subject']['groupIx'],
                       config['session']['sessionIx']]
    timeColumnsSess = ['sessDate',
                       'sessTime']
    timeDataSess    = [config['session']['date'],
                       config['session']['time']]
    # Define columns with data that does vary across trials
    # =========================================================================
    idColumns               = ['blockId']
    ixColumns               = ['blockIx',
                               'trialIx']
    timeColumns             = ['tSession',
                               'tBlock',
                               'trialOns',
                               'trialDur']
    feedbackColumns         = ['trialType',
                               'feedback']

    if config['stimConfig']['fix']['content'] is not None:
        ixColumns += ['fixIx']
        timeColumns+= ['fixOns','fixOnsDt','fixDur','fixDurDt']
    if config['stimConfig']['cue']['content'] is not None:
        ixColumns += ['cueIx']
        timeColumns += ['cueOns','cueOnsDt','cueDur','cueDurDt']
    if config['stimConfig']['s1']['content'] is not None:
        ixColumns += ['s1Ix']
        timeColumns += ['s1Ons','s1OnsDt','s1Dur','s1DurDt']
    if config['stimConfig']['s2']['content'] is not None:
        ixColumns += ['s2Ix','soaIx']
        timeColumns += ['s2Ons','s2OnsDt','s2Dur','s2DurDt']

    # Response event times
    # -------------------------------------------------------------------------
    # e.g. 'keyCount_f'
    # e.g. 'keyTime1_f', 'keyTime2_f'

    rspKeys = [item
               for sublist
               in config['apparatus']['rd']['settings']['rspKeys']
               for item in sublist]

    respEvColumns = list(chain.from_iterable(('keyCount_'+key,
                                              'keyTime1_'+key,
                                              'keyTime2_'+key)
                                             for key in rspKeys))


    # Descriptive stats
    # -------------------------------------------------------------------------
    trialStats      = config['statistics']['trial']
    rspKeyPairs    = config['apparatus']['rd']['settings']['rspKeys']
    statsColumns    = []

    if config['statistics']['trial']['rt']:
        statsColumns += list(chain.from_iterable(('rt1_'+key,
                                                  'rt2_'+key)
                                                 for key in rspKeys))
    if config['statistics']['trial']['rtDiff']:
        statsColumns += ['rtDiff1_'+pair[0]+'-'+pair[1] for pair in rspKeyPairs]
        statsColumns += ['rtDiff2_'+pair[0]+'-'+pair[1] for pair in rspKeyPairs]

    if config['statistics']['trial']['rpt']:
        statsColumns += list(chain.from_iterable(('rpt1_'+key,
                                                  'rpt2_'+key)
                                                 for key in rspKeys))



    columns += idColumnsSess
    columns += idColumns
    columns += ixColumnsSess
    columns += ixColumns
    columns += timeColumnsSess
    columns += timeColumns
    columns += respEvColumns
    columns += statsColumns
    columns += feedbackColumns
    log = DataFrame(index = index, columns = columns)

    # Add values for columns that remain stationary across the session
    sessColumns     = []
    sessColumns     += idColumnsSess
    sessColumns     += ixColumnsSess
    sessColumns     += timeColumnsSess
    sessData        = []
    sessData        += idDataSess
    sessData        += ixDataSess
    sessData        += timeDataSess

    return log, sessColumns, sessData

def init_config(runtime,configDir):
    """
    Parse and process experiment configuration

    :return config:
    """

    config = runtime.getConfiguration()

    userDefParams = runtime.getUserDefinedParameters()

    hub = runtime.hub

    ###########################################################################
    # STUDY
    ###########################################################################

    config['study'] = {'studyId':           config['code'],
                       'ethicsProtocolId':  config['ethics_protocol_id']
                       }

    ###########################################################################
    # SUBJECT
    ###########################################################################
    # subjectIx
    # groupIx
    # sex
    # age

    config['subject'] = {'subjectIx':   int(userDefParams['subjectIx']),
                         'groupIx':     int(userDefParams['groupIx']),
                         'sex':         userDefParams['subject_sex'],
                         'age':         userDefParams['subject_age']
                         }

    ###########################################################################
    # SESSION
    ###########################################################################

    now = time.gmtime(calendar.timegm(time.gmtime()) - core.getTime())

    config['session'] = {'sessionIx': int(userDefParams['sessionIx']),
                         'experimenterId': config['session_defaults']['experimenterId'],
                         'date': time.strftime('%Y-%m-%d',now),
                         'time': time.strftime('%H%M-GMT',now)
                         }

    ###########################################################################
    # APPARATUS
    ###########################################################################

    config['apparatus'] = {'hub':       hub,
                           'display':   dict(),
                           'kb':        dict(),
                           'rd':        dict()
                           }

    config['apparatus']['display']['client'] = hub.getDevice('display')
    print 'Display: %s' % config['apparatus']['display']['client']

    config['apparatus']['kb']['client'] = hub.getDevice('keyboard')
    print 'Keyboard: %s' % config['apparatus']['kb']['client']


    if hub.getDevice('responsedevice') is None:
        config['apparatus']['rd']['client'] = hub.getDevice('keyboard')
    else:
        config['apparatus']['rd']['client'] = hub.getDevice('responsedevice')
    print 'Response device: %s' % config['apparatus']['rd']['client']

    # Keyboard settings
    # -------------------------------------------------------------------------
    escKeys = config['responses']['abortKeys']
    toggleKeys = config['responses']['toggleKeys']
    config['apparatus']['kb']['settings'] = {'escKeys': escKeys,
                                             'toggleKeys': toggleKeys}

    kb = config['apparatus']['kb']['client']

    print 'Keyboard: \n'
    print '-------------------------------------------------------------------'
    print 'Reports events: %s' % kb.isReportingEvents()
    print 'Configuration: \n'
    pprint(kb.getConfiguration())

    # Response device settings
    # -------------------------------------------------------------------------
    rd = config['apparatus']['rd']['client']
    rdClass = rd.getIOHubDeviceClass()
    rspKeysPerClass = config['responses']['responseKeysPerClass']
    keyKey = {'Keyboard':   'key',
              'Serial':     'data'}
    timeKey = {'Keyboard':  'time',
               'Serial':    'time'}

    config['apparatus']['rd']['settings'] = {'class': rdClass,
                                             'rspKeys': rspKeysPerClass[rdClass],
                                             'keyKey': keyKey[rdClass],
                                             'timeKey': timeKey[rdClass]}

    # Enable event reporting and clear all recorded events
    rd.enableEventReporting()
    rd.clearEvents()

    print 'Response device : \n'
    print '-----'
    print 'Reports events: %s \n' % rd.isReportingEvents()
    print 'Configuration: \n'
    pprint(rd.getConfiguration())


    ###########################################################################
    # WINDOW
    ###########################################################################
    display = config['apparatus']['display']['client']

    window  = visual.Window(display.getPixelResolution(),
                            monitor = display.getPsychopyMonitorName(),
                            units = 'deg',
                            fullscr = False,
                            allowGUI = False,
                            screen = display.getIndex())

    frameRate = window.getActualFrameRate()
    frameTime = 1/frameRate
    config['window'] = {'window': window,
                        'frameRate': frameRate,
                        'frameTime': frameTime}

    ###########################################################################
    # STIMULI
    ###########################################################################

    # Make stimuli
    config['stimuli'] = {stim: define_stimulus(window,config['stimConfig'][stim])
                         for stim in config['stimConfig']
                         if config['stimConfig'][stim]['content'] is not None}

    ###########################################################################
    # RESPONSES
    ###########################################################################


    ###########################################################################
    # EVALUATION
    ###########################################################################
    trialEvalData = pd.read_csv(config['evaluation']['trial']['evalDataFile'][rdClass])



    trialCategories = trialEvalData.fillna(value=np.nan)

    evalColumns = [col for col in trialEvalData.columns
                   if not col.startswith('trial')]

    config['evaluation']['trial']['evalData'] = trialEvalData[evalColumns].copy()
    config['evaluation']['trial']['label'] = trialEvalData['trialLabelAbbrev'].copy()
    config['evaluation']['trial']['feedback'] = trialEvalData['trialFeedback'].copy()

    ###########################################################################
    # INSTRUCTION
    ###########################################################################

    config['stimuli']['instruction'] = {type: define_stimulus(window,
                                                              config['instruction'][type])
                                        for type in config['instruction']
                                        if config['instruction'][type]['content'] is not None}

    config['instruction']['practice']['list'] = pd.read_csv(config['instruction']['practice']['instructionListFile'])
    config['instruction']['experiment']['list'] = pd.read_csv(config['instruction']['experiment']['instructionListFile'])

    ###########################################################################
    # PRACTICE
    ###########################################################################

    # If a trialListFile exists, use this
    trialListFile = config['practice']['trialListFile']

    if os.path.isfile(trialListFile):
        trialList = pd.read_csv(trialListFile)
        ixCols = [col for col in trialList if re.search('Ix$',col)]


        # Assertions
        # ---------------------------------------------------------------------
        for col in ixCols:
            assert trialList[col].dtype == np.int or \
                   all(trialList['cueIx'].isnull()), \
                'column {} in file {} contains data other than integers'.format(col,trialListFile)



        config['practice']['trialList'] = trialList

    ###########################################################################
    # EXPERIMENT
    ###########################################################################

    # If a trialListFile exists, use this
    trialListFile = config['experiment']['trialListFile']

    if os.path.isfile(trialListFile):
        config['experiment']['trialList'] = pd.read_csv(trialListFile)

    ###########################################################################
    # PROCEDURE
    ###########################################################################

    # Clocks
    # -------------------------------------------------------------------------
    # clock = {'trial': core.Clock(),     # Reset once per trial, at no-signal onset
    #          'block': core.Clock(),     # Reset once per block, at block onset
    #          'session': core.Clock()}   # Reset once per session, at start of experiment

    # Stimulus-onset asynchronies
    # -------------------------------------------------------------------------
    # soaType = []

    # soa = {'type': soaType}

    # Feedback
    # -------------------------------------------------------------------------


    # Computation
    # =========================================================================


    # Evaluation
    # =========================================================================

    # Trial categories
    # -------------------------------------------------------------------------
    # trialCategories = pd.read_csv(config['evaluation']['trial']['file'])
    # trialCategories = trialCategories.fillna(value='NULL')
    # config['evaluation']['trial']['categories'] = trialCategories

    # Log
    # =========================================================================

    # Make a log directory, if it does not exist
    exptDir = os.path.abspath(os.path.join(configDir, os.pardir))
    logDir = os.path.normcase(os.path.join(exptDir,'log'))

    if not os.path.isdir(logDir):
        os.mkdir(logDir)

    # Run time info
    # -------------------------------------------------------------------------

    if config['log']['runtime']:

        runTimeInfo = info.RunTimeInfo(win=window,
                                       refreshTest=True,
                                       verbose=True,
                                       userProcsDetailed=True)

        strFormat = '%s_Study_%s_Group_%.2d_Subject_%.3d_Session_%.2d_%s_%s'

        fileName = strFormat % ('runTimeInfo',
                                config['study']['studyId'],
                                config['subject']['groupIx'],
                                config['subject']['subjectIx'],
                                config['session']['sessionIx'],
                                config['session']['date'],
                                config['session']['time'])


        filePath = os.path.normcase(os.path.join(logDir, fileName + '.csv'))

        with open(filePath,'wt') as fileObj:
            fileObj.write(str(runTimeInfo))

    # Task performance
    # -------------------------------------------------------------------------

    # Init log data frame
    performanceLog, sessColumns, sessData = init_log(config)

    config['log']['performance']['dataframe']   = performanceLog
    config['log']['performance']['sessColumns'] = sessColumns
    config['log']['performance']['sessData']    = sessData

    if config['log']['performance']['trial']:

        strFormat = '%s_Study_%s_Group_%.2d_Subject_%.3d_Session_%.2d_%s_%s'

        fileName = strFormat % ('taskPerformance',
                                config['study']['studyId'],
                                config['subject']['groupIx'],
                                config['subject']['subjectIx'],
                                config['session']['sessionIx'],
                                config['session']['date'],
                                config['session']['time'])

        filePath = os.path.normcase(os.path.join(logDir, fileName + '.csv'))

        config['log']['performance']['trialFile'] = filePath

        # Init log file on disk
        with open(filePath,'wt') as fileObj:
            config['log']['performance']['dataframe'].to_csv(fileObj, index=False, header=True)

    # Assemble procedure dictionary
    # -------------------------------------------------------------------------
    # procedure = {'clock': clock,
    #              'feedback': feedback,
    #              'soa': soa,
    #              'computation': computation,
    #              'log': log}

    # PUT EVERYTHING TOGETHER
    # =========================================================================
    # SPY = {'study': study,
    #        'subject': subject,
    #        'session': session,
    #        'files': files,
    #        'io': io,
    #        'apparatus': apparatus,
    #        'stimuli': stimuli,
    #        'window': window,
    #        'procedure': procedure}
    #
    # return SPY

    return config

def init_stimulus(window,stimType):
    """

    :return stimObject:
    """

    stimDict = {'textstim': visual.TextStim(window),
                'imagestim': visual.ImageStim(window)}

    stimObject = stimDict[stimType.lower()]

    return stimObject

def load_trial_list(file):
    """
    Read .csv file as pandas data frame

    :param file: path to trial list .csv file
    :return trialList: trial list as pandas data frame
    """

    trialList = pd.read_csv(file)

    return trialList

def log_trial():
    trialLog = []

    return trialLog

def make_log_file():
    """

    :return:
    """

    # Identify headers in trial list

    # Identify headers in log list

    # Fill values in log list with values in trial list

    # Return log file?

def make_trial_list():
    """ Summary line

    Extended description

    Arguments:
        arg1 (int): Description of arg1

    Returns:
        bool: Description of return value
    """

    trialList = [
        {'onsCue':      [],
         'onsS1':       [],
         'onsS2':       [],

         'durCue':      [],
         'durS1':       [],
         'durS2':       [],

         'respID':      [],
         'respTime':    []
         }
    ]

    return trialList

def get_flip_time():
    pass

def present_instruction(config,type,*args):
    """

    :param config:
    :param type:
    :param kwargs:
    :return:

    """

    # Process inputs
    # -------------------------------------------------------------------------

    # Process variable arguments
    if len(args) > 0:
        blockIx = args[0]


    window          = config['window']['window']
    hub             = config['apparatus']['hub']
    rd              = config['apparatus']['rd']
    kb              = config['apparatus']['kb']
    toggleKeys      = kb['settings']['toggleKeys']
    instructionStim = config['stimuli']['instruction'][type]

    stimIx = 0

    # Determine which instruction screens to show, depending on the experiment phase
    if type == 'practice' or type == 'experiment':
        sessionIx       = config['session']['sessionIx']
        instructionList = config['instruction'][type]['list']

        pattern         = {'sessionIx': [sessionIx],
                           'blockIx':   [blockIx]}
        ix              = instructionList.index.tolist()
        iRow            = instructionList[pattern.keys()].isin(pattern).all(1)
        stimList        = instructionList.loc[iRow,'instructionIx'].astype(int).tolist()
    elif type == 'start' or type == 'end':
        stimList        = range(len(instructionStim))


    # Show instruction screens and monitor keyboard and response device inputs
    # -------------------------------------------------------------------------

    while stimIx < len(stimList):

        # Clear buffer
        hub.clearEvents('all')
        rd['client'].clearEvents()
        kb['client'].clearEvents()
        noKeyPressed = True

        # Show the instruction
        instructionStim[stimList[stimIx]].draw()
        window.flip()

        while noKeyPressed:

            # Collect responses
            rdKeyCount, toggleKeyPressed = collect_response(rd,kb)

            # If user pressed key move to next stimulus
            if sum(rdKeyCount.values()) > 0:
                stimIx += 1
                break

            # If toggle keys are used move to next or previous stimulus
            if toggleKeyPressed:
                if toggleKeyPressed == toggleKeys[0]:
                    stimIx -= 1
                    if stimIx < 0:
                        stimIx = 0
                    break
                elif toggleKeyPressed == toggleKeys[1]:
                    stimIx += 1
                    break

def present_stimuli(window,stimList,u,f_on_off,log,timing):
    """


    What is logged?
    - onset and duration of trial (trialOns, trialDur), relative to core.getTime()
    - onset and duration of all stimuli (e.g. fixOns, fixDur), relative to trial onset, in seconds


    :param window:
    :param stimList:
    :param u:
    :param f_on_off:
    :param log:
    :return: trial
    """

    # Define dynamic variables
    # -------------------------------------------------------------------------
    nStim = len(stimList)
    nFrame = np.size(u,1)
    tFlip = [None] * (nFrame + 1)

    # Execute trial
    # =========================================================================

    # Draw stimuli, flip screen
    for frameIx in range(nFrame):

        for stimIx in range(nStim -1):

            if u[stimIx,frameIx]:
                stimList[stimIx].setAutoDraw(True)
            else:
                stimList[stimIx].setAutoDraw(False)

        while core.getTime() < (timing['ons'] +
                                timing['dur'] +
                                timing['ITIDur'] -
                                timing['refreshTime']):
            pass

        tFlip[frameIx] = window.flip()

    # Hide all trial stimuli and present ITI stimulus
    [stimList[stimIx].setAutoDraw(False) for stimIx in range(nStim - 1)]
    stimList[-1].setAutoDraw(True)
    tFlip[-1] = window.flip()

    # Timing
    trialOns = tFlip[0]
    trialOff = tFlip[-1]
    trialDur = trialOff - trialOns

    # print 'trialOns = %.3f s' % trialOns

    # Actual stimulus onset and duration times
    stimDisplayed = [stim.name for stim in stimList]

    # Log
    log.iloc[0]['trialOns'] = trialOns
    log.iloc[0]['trialDur'] = trialDur

    for ix in range(len(stimDisplayed) - 1):
        fOn,fOff    = f_on_off[:,ix]

        ons         = tFlip[fOn] - trialOns
        dur         = tFlip[fOff] - tFlip[fOn]
        onsIntended = log.iloc[0][stimDisplayed[ix]+'Ons']
        durIntended = log.iloc[0][stimDisplayed[ix]+'Dur']
        onsDt       = ons - onsIntended
        durDt       = dur - durIntended

        log.iloc[0][stimDisplayed[ix]+'Ons'] = ons
        log.iloc[0][stimDisplayed[ix]+'Dur'] = dur
        log.iloc[0][stimDisplayed[ix]+'OnsDt'] = onsDt
        log.iloc[0][stimDisplayed[ix]+'DurDt'] = durDt

    return log

def run_block(config,trialList,blockId,performanceLog):
    """ Summary line

    Extended description

    :var window:
    :var stimuli:
    :param iBlock: block index


    """

    # 1. Process inputs
    # =========================================================================

    # All relevant variables and objects
    window          = config['window']['window']
    stimuli         = config['stimuli']
    hub             = config['apparatus']['hub']
    rd              = config['apparatus']['rd']
    kb              = config['apparatus']['kb']
    trialStats      = config['statistics']['trial']

    trialEvalData   = config['evaluation']['trial']

    sessColumns     = config['log']['performance']['sessColumns']
    sessData        = config['log']['performance']['sessData']

    blockIx         = trialList.iloc[0]['blockIx']

    performanceLogColumns = performanceLog.columns

    # Define dynamic variables
    # -------------------------------------------------------------------------

    # Mock times, so that trial starts immediately
    trialTiming = {'ons':           -float('inf'),
                   'dur':           0,
                   'ITIDur':        0,
                   'refreshTime':   1/config['window']['frameRate']}
    blockOns = -float('inf')

    # =========================================================================
    trialIxList = trialList.index.tolist()


    # Present trials
    # =========================================================================

    for trialIx in trialIxList:

        # Prepare trial
        # ---------------------------------------------------------------------
        trialLog = DataFrame(index = [trialIx],
                             columns = performanceLogColumns)

        # Fill in trial-independent data
        # ---------------------------------------------------------------------
        trialLog.loc[trialIx,sessColumns] = sessData

        trialLog, stimList, u, f_on_off, t_max = stim_to_frame_mat(config,trialList.ix[trialIx],trialLog)

        # Clear buffer
        # ---------------------------------------------------------------------
        hub.clearEvents('all')
        rd['client'].clearEvents()
        kb['client'].clearEvents()

        # Run trial
        # ---------------------------------------------------------------------
        trialLog    = run_trial(config,
                                trialLog,
                                trialTiming,
                                window,
                                stimList,
                                u,
                                f_on_off,
                                rd,
                                kb,
                                trialStats,
                                trialEvalData,
                                stimuli)

        # print 'Time since previous trial: %.3f s' % (trialLog['trialOns'].item() - trialTiming['ons'])

        # Update timings
        trialTiming['ons']      = trialLog['trialOns'].item()
        trialTiming['dur']      = t_max
        trialTiming['ITIDur']   = trialList.ix[trialIx]['ITIDur']

        # Log trial data not logged inside run_trial
        # ---------------------------------------------------------------------
        trialLog['blockId']     = blockId

        trialLog['blockIx']     = blockIx
        trialLog['trialIx']     = trialList.ix[trialIx]['trialIx']

        # Session timing
        sm, ss = divmod(trialTiming['ons'], 60)
        sh, sm = divmod(sm, 60)
        trialLog['tSession']    = '%d:%02d:%02d' % (sh, sm, ss)

        # Block timing
        if trialIx == trialIxList[0]:
            blockOns = trialTiming['ons']
        bs, bms = divmod(trialTiming['ons'] - blockOns,1)
        bm, bs = divmod(bs, 60)
        bh, bm = divmod(bm, 60)
        trialLog['tBlock']      = '%d:%02d:%02d.%03d' % (bh, bm, bs,bms*1000)

        # Append trial data to data frame and file
        # ---------------------------------------------------------------------
        performanceLog = performanceLog.append(trialLog)

        with open(config['log']['performance']['trialFile'],'a') as fileObj:
            trialLog.to_csv(fileObj, index=False, header=False, na_rep=np.nan)

    # Compute block stats
    # =========================================================================

    # If this is the right practice block, then compute SOAs and set them in
    # trialList of expt

    # Present block feedback and write data if necessary
    # -------------------------------------------------------------------------

    return performanceLog

def run_trial(config,trialLog,trialTiming,window,stimList,u,f_on_off,rd,kb,trialStats,trialEvalData,stimuli):


    # Present stimuli
    # -------------------------------------------------------------------------
    trialLog = present_stimuli(window=window,
                               stimList=stimList,
                               u=u,
                               f_on_off=f_on_off,
                               log=trialLog,
                               timing=trialTiming)

    # Collect responses
    # -------------------------------------------------------------------------
    trialLog = collect_response(rd=rd,
                                kb=kb,
                                log=trialLog)

    # Compute trial statistics
    # -------------------------------------------------------------------------
    trialLog = compute_trial_statistics(trialStats=trialStats,
                                        rd=rd,
                                        log=trialLog)

    # Evaluate trial
    # -------------------------------------------------------------------------
    trialLog = evaluate_trial(evalData=trialEvalData,
                              window=window,
                              stimuli=stimuli,
                              log=trialLog)

    # Wrap up
    # -------------------------------------------------------------------------
    return trialLog


def set_soa(config,log):

    # Process inputs, define variables
    # -------------------------------------------------------------------------
    fTime = config['window']['frameTime']

    soaIx = config['experiment']['trialList']['soaIx'].dropna().unique()
    nSoa = len(soaIx)

    # Identify no-signal trials
    iNS = performanceLog.s1Ix.notnull() & performanceLog.s2Ix.isnull()

    # Response time data from first response
    rtCols = [col for col in list(performanceLog) if col.startswith('rt1')]
    rtData = performanceLog[rtCols]
    meanRt = rtData.mean().mean()

    # Proportion steps
    propStep = {2: .75, # .15, .90 of meanRT
                3: .40, # .15, .55, .95 of meanRT
                4: .25, # .15, .40, .65, .90 of meanRT
                5: .20, # .15, .35, .55, .75, .95 of meanRT
                6: .15} # .15, .25, .35, .45, .55, .65, .75 of meanRT

    # Determine new SOAs
    soa1        = np.around(meanRt * .15 / fTime) * fTime
    soaStep     = np.around(meanRt * propStep(min([nSoa,6])) / fTime) * fTime
    soa         = np.linspace(soa1,soa1+(nSoa-1)*propStep,nSoa)

    # Update soa experiment trial list
    for ix in range(nSoa):
        pass





def time_to_frame(ons, dur, dt, t_max):
    """ Summary line

    Extended description

    Arguments:
        arg1 (int): Description of arg1

    Returns:
        bool: Description of return value
    """

    ###########################################################################
    # 1. PROCESS INPUTS & SPECIFY VARIABLES
    ###########################################################################

    # 1.1. Import libraries
    # =========================================================================


    # 1.2. Process inputs
    # =========================================================================
    # Check if all inputs are ndarrays

    assert isinstance(ons, np.ndarray), 'ons should be of type ndarray'
    assert isinstance(dur, np.ndarray), 'dur should be of type ndarray'
    assert isinstance(dt, np.ndarray), 'dt should be of type ndarray'
    assert isinstance(t_max, np.ndarray), 't_max should be of type ndarray'

    #TODO: assert that inputs are no unsized objects;

    # 1.3. Define dynamic variables
    # =========================================================================
    # Number of stimuli
    n_stim = np.size(ons)

    # Number of frames
    n_frame = int(np.ceil(t_max/dt))

    # Array of frames
    f = range(0, n_frame, 1)

    # Array of time points
    t = np.linspace(0, n_frame * float(dt), n_frame, True)

    # Stimulus onsets and offsets, in frames
    f_on = np.around(ons/dt).astype(int)
    f_off = np.around((ons + dur)/dt).astype(int)

    # Preallocate u
    u = np.zeros((n_stim, n_frame)).astype(int)

    ###########################################################################
    # 2. SPECIFY STIMULUS AND INPUT MATRICES
    ###########################################################################

    # Loop over stimuli
    for i_stim in range(0, n_stim):

        # Stimulus onsets in stimulus-frame diagram
        u[i_stim, f == f_on[i_stim]] += 1

        # Stimulus offsets in stimulus-frame diagram
        u[i_stim, f == f_off[i_stim]] -= 1

    # Encode 'stimulus on' as 1.0, 'stimulus off' as 0.0
    u = np.cumsum(u, axis=1)

    # Convert to boolean
    u = u.astype(bool)

    # Stimulus on- and offset in frames
    f_on_off = np.vstack([f_on,f_off])

    ########################################
    # 3. SPECIFY OUTPUT
    ########################################
    return u, f_on_off, t