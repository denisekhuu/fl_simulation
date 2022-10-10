
class ModelAggregator():
    def federated_averaging(self, parameters):
        new_params = {}
        for name in parameters[0].keys():
            new_params[name] = sum([param[name].data for param in parameters]) / len(parameters)
        return new_params