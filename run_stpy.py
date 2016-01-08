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
from stpy import init_config, present_instruction, run_block, set_soa
import sys
import os

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

        def run_phase(phaseName,config,log):

            phaseTag        = phaseName[0].capitalize()
            trialList       = config[phaseName]['trialList']
            blockList       = trialList['blockIx'].unique()

            for blockIx in blockList:

                print 'Run %s block %d' % (phaseName,blockIx)

                trialListBlock = trialList[trialList['blockIx'] == blockIx]

                # set_soa(config=config,
                #         log=log)

                present_instruction(config,'practice',blockIx)
                core.wait(1)

                performanceLog  = run_block(config=config,
                                            trialList = trialListBlock,
                                            blockId = phaseTag + str(blockIx),
                                            performanceLog=log)

        # Parse and process configuration
        # ---------------------------------------------------------------------
        modDir = args[0][1]
        config  = init_config(self,modDir)

        # Present welcome screen and general instruction
        # ---------------------------------------------------------------------
        present_instruction(config,'start')

        # Get performance log data frame
        # ---------------------------------------------------------------------
        performanceLog  = config['log']['performance']['dataframe']

        # Run practice blocks
        # ---------------------------------------------------------------------
        performanceLog = run_phase('practice',
                                   config=config,
                                   log=performanceLog)

        # Run experimental blocks
        # ---------------------------------------------------------------------
        performanceLog = run_phase('experiment',
                                   config=config,
                                   log=performanceLog)

        # Terminate experiment
        # ---------------------------------------------------------------------
        present_instruction(config,'end')
        core.wait(5)
        core.quit()

####### Main Script Launching Code Below #######

if __name__ == "__main__":
    import os
    from psychopy import gui
    from psychopy.iohub import module_directory

    def main(modDir):
        """

        :param modDir: Directory where experiment configuration file resides

        """

        # Let user select response device
        # ---------------------------------------------------------------------
        rdConfigFiles = {'Keyboard':
                            'response_device_config_files/keyboard_config.yaml',
                         'Serial':
                             'response_device_config_files/serial_config.yaml',
                         'fORP':
                             'response_device_config_files/forp_config.yaml'}

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
        baseConfigFile = os.path.normcase(os.path.join(modDir,
                                                       'iohub_config_part.yaml'))

        respDevConfigFile = os.path.normcase(os.path.join(modDir,
                                rdConfigFiles[dlg_info.values()[0]]))

        combinedConfigFile = os.path.normcase(os.path.join(modDir,
                                                           'iohub_config.yaml'))

        ExperimentRuntime.mergeConfigurationFiles(baseConfigFile,
                                                  respDevConfigFile,
                                                  combinedConfigFile)

        # Start the experiment
        # ---------------------------------------------------------------------
        runtime=ExperimentRuntime(modDir, "experiment_config.yaml")
        runtime.start((dlg_info.values()[0],modDir))

    # Get the current directory, using a method that does not rely on __FILE__
    # or the accuracy of the value of __FILE__.
    modDir = module_directory(main)

    # Run the main function, which starts the experiment runtime
    main(modDir)