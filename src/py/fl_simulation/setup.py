import os
from pathlib import Path
from torch import save

def create_default_model(config):
    """
    Create default model parameters as initialization for client and server models.
    This will make sure that the default behaviour is the same. 
    You will find the default model in the folder defined by configuration class with the following logic: 
    
    path: {working_dir}/{config.TEMP}/models/, "{config.MODELNAME}.model

    Parameters
    ------
    :param config : Configurations regarding the CNN model
    :type config: Configurations 
    
    :return:
    ------
    
    
    Configuration Object needs the following attributes
    
    Attributes 
    ------
    MODELNAME : string 
    Name of the model which defines the name of the file
    
    NETWORK : nn.Module
    CNN model definition
    
    TEMP : string 
    Name of temporary directory with all temporary files 

    """
    
    default_model_path = os.path.join(config.cwd, config.TEMP, 'models', "{}.model".format(config.MODELNAME))
    net = config.NETWORK()
    save(net.state_dict(), default_model_path)
    config.root.info("Default Model for {} saved to: {}".format(config.MODELNAME, os.path.dirname(default_model_path)))