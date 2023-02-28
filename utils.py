import argparse
import configparser as ConfigParser 
import ast
import torch

def parse_args():
    """
        Parse the command line arguments
        """
    parser = argparse.ArgumentParser()
    parser.add_argument('-C','--config', default="config_bbb.txt", required=True, help='Name of the input config file')
    
    args, __ = parser.parse_known_args()
    
    return vars(args)

# -----------------------------------------------------------

def parse_config(filename):
    
    config = ConfigParser.ConfigParser(allow_no_value=True)
    config.read(filename)
    
    # Build a nested dictionary with tasknames at the top level
    # and parameter values one level down.
    taskvals = dict()
    for section in config.sections():
        
        if section not in taskvals:
            taskvals[section] = dict()
        
        for option in config.options(section):
            # Evaluate to the right type()
            try:
                taskvals[section][option] = ast.literal_eval(config.get(section, option))
            except (ValueError,SyntaxError):
                err = "Cannot format field '{0}' in config file '{1}'".format(option,filename)
                err += ", which is currently set to {0}. Ensure strings are in 'quotes'.".format(config.get(section, option))
                raise ValueError(err)

    return taskvals, config



def flatten(model):
    return torch.nn.utils.parameters_to_vector(model.parameters())

def unflatten(params, model):
    
    torch.nn.utils.vector_to_parameters(params, model.parameters()) 
    return None

# def flatten(model):
#     return torch.cat([p.flatten() for p in model.parameters()]).requires_grad_()


# def unflatten(model, flattened_params):
#     if flattened_params.dim() != 1:
#         raise ValueError('Expecting a 1d flattened_params')
#     params_list = []
#     i = 0
#     for val in list(model.parameters()):
#         length = val.nelement()
#         param = flattened_params[i:i+length].view_as(val)
#         params_list.append(param)
#         i += length
    
#     #params_tensor = torch.tensor(params_list, dtype=torch.float32)

#     return params_list#params_tensor