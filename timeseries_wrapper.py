from pandapower import timeseries as ts
from pandapipes.timeseries import run_timeseries
import pandas as pd 

class TimeseriesWrapper:
    def __init__(self, net, output_writer : ts.OutputWriter, log_variables) -> None:
        self.net = net
        self.output_writer = output_writer
        self.log_variables = log_variables

        # set up output dictionary
        self.output = { f"{table}.{variable}": None for table, variable in self.log_variables}

    def run_timeseries(self, net, time_steps):
        """runs a time series simulation and takes care of the output writer's
        data frames

        Args:
            net (pp.pandapipesNet): a pandapipes net to be simulated
            time_steps (range[int]): an iterable (assumed to be a range)
        """
        output_frames = { f"{table}.{variable}":[] for table, variable in self.log_variables}
        for t in time_steps:
            run_timeseries(net, time_steps=range(t, t + 1))
            # update all the data frames the output writer writes to 
            for table, variable in self.log_variables:
                key = f"{table}.{variable}"
                output_frames[key].append(self.output_writer.output[key])

        # finally convert all of them to dataframes again
        for table, variable in self.log_variables: 
            key = f"{table}.{variable}"
            self.output[key] = pd.concat(output_frames[key])

    def run_timestep(self, net, time_step):
        run_timeseries(net, time_steps=range(time_step, time_step+1))
        # add the recent data frame to the previous ones
        for table, variable in self.log_variables: 
            key = f"{table}.{variable}"
            if self.output[key] is None: # not been set yet
                self.output[key] = self.output_writer.output[key]
            else: 
                self.output[key] = pd.concat( [self.output[key], self.output_writer.output[key]], ignore_index=True)
        pass