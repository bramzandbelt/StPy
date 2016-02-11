# !/Users/bramzandbelt/anaconda/envs/psychopyenv/bin python
# -*- coding: utf-8 -*-

"""
Module docstring

"""

__author__      = 'bramzandbelt (bramzandbelt@gmail.com)'
__copyright__   = 'Copyright (c) 2015 Bram Zandbelt'
__license__     = 'CC BY 4.0'
__version__     = '0.1'
__vcs_id__      = ''

###############################################################################
# IMPORT MODULES
###############################################################################
from psychopy import core
from psychopy.iohub import ioHubExperimentRuntime
from stpy import init_config, present_instruction, run_phase
import sys
import pandas as pd
import glob

# Assure that text files are read in as Unicode (utf8) instead of ASCII
reload(sys)
sys.setdefaultencoding('utf8')

class ExperimentRuntime(ioHubExperimentRuntime):
    """

    Create class

    """

    def run(self,*args):
        """

        The run method contains your experiment logic. It is equal to what would be in your main psychopy experiment
        script.py file in a standard psychopy experiment setup. That is all there is too it really.

        :param args:

        """
        # from psychopy import gui

        # Parse and process configuration
        # ---------------------------------------------------------------------
        modDir = args[0][1]
        config  = init_config(self,modDir)

        sessionIx   = config['session']['sessionIx']

        # Determine if data will be overwritten
        trialLogFile = config['log']['performance']['trial']['file']

        if os.path.isfile(trialLogFile):
            groupIx     = config['subject']['groupIx']
            subjectIx   = config['subject']['subjectIx']


            trialLog = pd.read_csv(trialLogFile)

            if sessionIx in trialLog.sessionIx:
                warnDlg = gui.Dlg(title="WARNING",
                                  labelButtonOK=u' Continue ',
                                  labelButtonCancel=u' Cancel ')
                warnDlg.addText('You specified the following settings:')
                warnDlg.addFixedField('Group index:',groupIx)
                warnDlg.addFixedField('Subject index:', subjectIx)
                warnDlg.addFixedField('Session index:',sessionIx)
                warnDlg.addText('')
                warnDlg.addText('You might have entered the wrong data. A log file with these data already exists:')
                warnDlg.addText(trialLogFile)
                warnDlg.addText('')
                warnDlg.addText('Press Continue if you want to use the above settings and overwrite/append this file.')
                warnDlg.addText('Press Cancel if you want to change settings.')

                warnDlg.show()

                if not warnDlg.OK:
                    return -1

        # Present welcome screen and general instruction
        # ---------------------------------------------------------------------
        present_instruction(config,'start')

        # Run practice blocks
        # ---------------------------------------------------------------------
        if config['practice']['enable']:

            pTrialList      = config['practice']['trialList']
            pTrialList      = pTrialList[pTrialList.sessionIx == sessionIx]

            run_phase(config=config,
                      phaseId='practice',
                      trialList=pTrialList)

        # Run experimental blocks
        # ---------------------------------------------------------------------
        if config['experiment']['enable']:

            eTrialList      = config['experiment']['trialList']
            eTrialList      = eTrialList[eTrialList.sessionIx == sessionIx]

            run_phase(config=config,
                      phaseId='experiment',
                      trialList=eTrialList)

        # Terminate experiment
        # ---------------------------------------------------------------------
        present_instruction(config,'end')
        core.quit()

####### Main Script Launching Code Below #######

if __name__ == "__main__":
    import os
    from psychopy import gui
    from psychopy.iohub import module_directory

    def main(modDir):
        """

        :param modDir: Directory where module files reside

        """

        configDir = os.path.normcase(os.path.join(modDir,'config/'))

        # Let user select response device
        # ---------------------------------------------------------------------
        rdConfigFiles = {'Keyboard':
                            'iohub_keyboard.yaml',
                         'Serial':
                             'iohub_serial.yaml',
                         'fORP':
                             'iohub_forp.yaml'}

        info = {'Response Device = ': ['Select',
                                       'Keyboard',
                                       'Serial',
                                       'fORP']}

        dlg_info=dict(info)
        infoDlg = gui.DlgFromDict(dictionary = dlg_info,
                                  title = 'Select response device')
        if not infoDlg.OK:
            return -1

        while dlg_info.values()[0] == u'Select' and infoDlg.OK:
                dlg_info = dict(info)
                infoDlg = gui.DlgFromDict(dictionary=dlg_info,
                                          title='SELECT Response device to continue...')

        if not infoDlg.OK:
            return -1

        # Merge iohub configuration files
        # ---------------------------------------------------------------------
        baseConfigFile = os.path.normcase(os.path.join(configDir,
                                                       'iohub_base.yaml'))

        respDevConfigFile = os.path.normcase(os.path.join(configDir,
                                rdConfigFiles[dlg_info.values()[0]]))

        combinedConfigFile = os.path.normcase(os.path.join(configDir,
                                                           'iohub_config.yaml'))

        ExperimentRuntime.mergeConfigurationFiles(baseConfigFile,
                                                  respDevConfigFile,
                                                  combinedConfigFile)

        # Determine which experiment configuration file to use
        # ---------------------------------------------------------------------
        os.chdir(configDir)
        exptFiles = ['Select']
        exptFiles.extend(glob.glob('./expt_*.yaml'))
        os.chdir(modDir)

        exptConfigInfo = {'Experiment config file = ': exptFiles}
        exptConfigInfo = dict(exptConfigInfo)

        exptConfigDlg = gui.DlgFromDict(dictionary = exptConfigInfo,
                                        title = 'Select experiment config file')
        if not exptConfigDlg.OK:
            return -1

        while exptConfigInfo.values()[0] == u'Select' and exptConfigDlg.OK:
            exptConfigInfo = dict(exptConfigInfo)
            exptConfigDlg = gui.DlgFromDict(dictionary=exptConfigInfo,
                                          title='SELECT experiment config file to continue...')

        if not exptConfigDlg.OK:
            return -1

        # Start the experiment
        # ---------------------------------------------------------------------
        runtime=ExperimentRuntime(configDir, exptConfigInfo.values()[0])
        runtime.start((dlg_info.values()[0],modDir))

    # Get the current directory, using a method that does not rely on __FILE__
    # or the accuracy of the value of __FILE__.
    modDir = module_directory(main)

    # Run the main function, which starts the experiment runtime
    main(modDir)