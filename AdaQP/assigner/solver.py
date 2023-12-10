import time
import logging
import torch
import os
from typing import Dict, List, Tuple, Union
from itertools import chain
from multiprocessing.pool import ThreadPool
from queue import Queue
from torch import Tensor
import numpy as np
import pulp as plp
import pickle
import inotify_simple
import threading
import shutil

#from .profile import *
#from ..helper import BitType
#from ..communicator import BITS_SET
#from ..communicator import Communicator as comm
#from ..manager import GraphEngine as engine

logger = logging.getLogger('trainer')

# Function to be executed when a new file is added
def on_new_file(event, first_time = False):
    #print(f"New file added: {event.name}")
    #layer_key = event.name
    print(f"New file added: {event}")
    layer_key = event
    #os.remove("" + layer_key)
    bits_set = pickle.load(open("" + layer_key + "_bits_set.dat", "rb"))
    var_matrix = pickle.load(open("" + layer_key + "_var_matrix.dat", "rb"))
    comm_matrix = pickle.load(open("" + layer_key + "_comm_matrix.dat", "rb"))
    cost_model = pickle.load(open("" + layer_key + "_cost_model.dat", "rb"))
    coe_lambda = pickle.load(open("" + layer_key + "_coe_lambda.dat", "rb"))
    world_size = pickle.load(open("" + layer_key + "_world_size.dat", "rb"))
    normal_mode = pickle.load(open("" + layer_key + "_normal_mode.dat", "rb"))
    get_solution(bits_set, var_matrix, comm_matrix, cost_model, coe_lambda, world_size, normal_mode, layer_key, first_time)


ASSIGNMENT_SCHEME = ('uniform', 'random', 'adaptive')

def get_solution(bits_set, var_matrix: Dict[str, Dict[str, np.ndarray]], comm_matrix: Dict[str, Dict[str, np.ndarray]], cost_model: Dict[str, np.ndarray], coe_lambda: float, world_size: int, normal_mode: str = 'nadir_utopia', layer_key: str = '', first_time: bool = False):
    '''
    invode solver to get the solution.
    '''

    def get_scaling_factor(var_matrix: Dict[str, Dict[str, np.ndarray]], comm_matrix: Dict[str, Dict[str, np.ndarray]], cost_model: Dict[str, np.ndarray], normal_mode: str, world_size: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        # magnitude normalization
        if normal_mode == 'magnitude':
            # get magnitude for variance objective
            max_var = 0
            for _, data_per_channel in var_matrix.items():
                max_var += sum(data_per_channel[0])  # all groups all assigned with 2 bits
            max_comm_time = 0
            # find max comm time in each round and sum the max comm time in all rounds to get the magnitude
            # from round 1 -> round world_size - 1
            for round in range(1, world_size):
                round_max_comm_time = 0
                for rank in range(world_size):
                    dst = (rank + round) % world_size
                    channel_key = f'{rank}_{dst}'
                    max_current_channel = cost_model[channel_key][0] * sum(comm_matrix[channel_key][-1]) + cost_model[channel_key][1]
                    if round_max_comm_time < max_current_channel:
                        round_max_comm_time = max_current_channel  # all groups are assigned with 8 bits
                max_comm_time += round_max_comm_time
            return max_var, max_comm_time
        else:
            # get nadir and utopia solutions for variance objective
            var_nadir, var_utopia = 0.0, 0.0
            for _, data_per_channel in var_matrix.items():
                var_nadir += sum(data_per_channel[0])  # maximum variance
                var_utopia += sum(data_per_channel[-1])  # minimum variance
            # get nadir and utopia values for time objective
            time_nadir, time_utopia = 0.0, 0.0
            # get nadir and utopia values for each communication round
            # from round 1 -> round world_size - 1
            for round in range(1, world_size):
                round_time_nadir, round_time_utopia = float('-inf'), float('inf')
                for rank in range(world_size):
                    dst = (rank + round) % world_size
                    channel_key = f'{rank}_{dst}'
                    # all groups are assigned with 8 bits
                    nadir_current_channel = cost_model[channel_key][0] * sum(comm_matrix[channel_key][-1]) + cost_model[channel_key][1]
                    # all groups are assigned with 2 bits
                    utopia_current_channel = cost_model[channel_key][0] * sum(comm_matrix[channel_key][0]) + cost_model[channel_key][1]
                    if round_time_nadir < nadir_current_channel:
                        round_time_nadir = nadir_current_channel
                    if round_time_utopia > utopia_current_channel:
                        round_time_utopia = utopia_current_channel
                time_nadir += round_time_nadir
                time_utopia += round_time_utopia
            return (var_nadir, var_utopia), (time_nadir, time_utopia)

    def add_constraint(opt_model: plp.LpProblem, var_comm_vars: Dict[Tuple[int, int], plp.LpVariable], comm_matrix: Dict[str, Dict[str, np.ndarray]], Z: List[plp.LpVariable], cost_model: Dict[str, np.ndarray], size_buffer: Dict[str, List[int]], world_size: int):
        # add constraint for auxillary variables [Z_i] for each comm round
        for round in range(1, world_size):
            for rank in range(world_size):
                dst = (rank + round) % world_size
                channel_key = f'{rank}_{dst}'
                data_per_channel = comm_matrix[channel_key]
                channel_cost_model = cost_model[channel_key]
                channel_vars = var_comm_vars[channel_key]
                channel_size = size_buffer[channel_key]
                channel_constraint = []
                channel_constraint.extend([channel_vars[i, j] * data_per_channel[i, j] * channel_cost_model[0] for i in range(channel_size[0]) for j in range(channel_size[1])])  # add the output comm cost of the cost model 
                channel_constraint.extend([channel_cost_model[1], -1 * Z[round - 1]])  # add latency and auxillary variable Z_i for round i
                opt_model.addConstraint(plp.LpConstraint(e=plp.lpSum(channel_constraint), sense=plp.LpConstraintLE, rhs=0))
        # add constraint for binary decision variables (each group can only be assigned with one type of bits)
        for channel_key, var_per_channel in var_comm_vars.items():
            size_channel = size_buffer[channel_key]
            for j in range(size_channel[1]):
                opt_model.addConstraint(plp.LpConstraint(e=plp.lpSum(var_per_channel[i, j] for i in range(size_channel[0])), sense=plp.LpConstraintEQ, rhs=1))

    # get normalization factor
    # assert normal_mode in self.normalization_mode, f'normalization mode {normal_mode} is not supported'
    scale_var, scale_time = get_scaling_factor(var_matrix, comm_matrix, cost_model, normal_mode, world_size)
    # init variables
    num_bits_options = len(bits_set)
    var_comm_vars: Dict[Tuple[int, int], plp.LpVariable] = {}
    size_buffer: Dict[str, List[int, int]] = {}
    for channel_key, data_per_channel in var_matrix.items():
        # set variables for current channel, x_{i, j} means allocate bits i to group j
        num_worker_option = data_per_channel.shape[-1]
        size_buffer[channel_key] = [num_bits_options, num_worker_option]
        vars = {(i, j): plp.LpVariable(cat=plp.LpBinary, name=f'{channel_key}_x_{i}_{j}') for i in range(num_bits_options) for j in range(num_worker_option)}
        var_comm_vars[channel_key] = vars
    # init auxillary variables Z_i for each round
    Z = [plp.LpVariable(cat=plp.LpContinuous, name=f'Z_{i}') for i in range(1, world_size)]
    # define the problem and add constraints
    opt_model = plp.LpProblem(name='MIP_Model', sense=plp.const.LpMinimize)
    add_constraint(opt_model, var_comm_vars, comm_matrix, Z, cost_model, size_buffer, world_size)
    # set objectives
    total_var = []
    for channel_key, data_per_channel in var_matrix.items():
        channel_var = var_comm_vars[channel_key]
        channel_size = size_buffer[channel_key]
        total_var.extend(channel_var[i, j] * data_per_channel[i, j] for i in range(channel_size[0]) for j in range(channel_size[1]))
    if normal_mode == 'magnitude':
        objective = coe_lambda * plp.lpSum(total_var) / scale_var + (1 - coe_lambda) * plp.lpSum(Z) / scale_time  # variance & time objectives Scalarization
    else:
        objective = coe_lambda * (plp.lpSum(total_var) - scale_var[-1]) / (scale_var[0] - scale_var[-1]) + (1 - coe_lambda) * (plp.lpSum(Z) - scale_time[-1]) / (scale_time[0] - scale_time[-1])
    opt_model.setObjective(objective)
    # solve the problem
    #available_solvers = plp.list_solvers(onlyAvailable=True)
    start = time.time()
    current_solver = os.getenv("SOLVER", "HiGHS_CMD")
    timeLimit = int(os.getenv("TIMELIMIT", "1000000"))
    if first_time == True:
        print(layer_key)
        print("No Early Stopping")
        timeLimit = 1000000
    #opt_model.to_json(str(time.time()) + ".json")
    #opt_model.writeLP(str(time.time()) + ".lp")
    #opt_model.writeMPS(str(time.time()) + ".mps")
    #print(available_solvers)
    if current_solver == "GUROBI":
        opt_model.solve(plp.GUROBI(msg=True, timeLimit=timeLimit))
    elif current_solver == "CPLEX_PY":
        opt_model.solve(plp.CPLEX_PY(msg=True, timeLimit=timeLimit))
    elif current_solver == "GLPK_CMD":
        #opt_model.writeMPS("test_model.mps")
        #var, opt_model = plp.LpProblem.fromMPS("test_model.mps")
        opt_model.solve(plp.GLPK_CMD(msg=True, timeLimit=timeLimit)) # for glp we'll just do this manually
    elif current_solver == "HiGHS_CMD":
        opt_model.solve(plp.HiGHS_CMD(msg=True, timeLimit=timeLimit))
    elif current_solver == "PULP_CBC_CMD":
        opt_model.solve(plp.PULP_CBC_CMD(msg=True, timeLimit=timeLimit))
    else:
        opt_model.solve(plp.PULP_CBC_CMD(msg=True, timeLimit=timeLimit))
    solving_time = time.time() - start
    # get the optimal solution
    bits_assignment = {}

    if opt_model.to_dict()['parameters']['sol_status'] == 2:
        solving_time = -1
        print("Early Stopping")
        pickle.dump(solving_time, open("" + layer_key + "_solving_time.dat", "wb"))
        pickle.dump(bits_assignment, open("" + layer_key + "_bits_assignment.dat", "wb"))
        pickle.dump(0, open("" + layer_key, "wb"))
        return

    for channel_key, data_per_channel in var_comm_vars.items():
        channel_size = size_buffer[channel_key]
        channel_rst = torch.zeros(channel_size[-1], dtype=torch.int32)
        x_vars_tensor = torch.tensor([x.value() for x in data_per_channel.values()]).view(channel_size[0], channel_size[1])  # [N, K]
        for i in range(x_vars_tensor.shape[0]):
            idx = torch.nonzero(x_vars_tensor[i])
            channel_rst[idx] = bits_set[i]  # the group rst
        bits_assignment[channel_key] = channel_rst
    #print(solving_time)
    #print(bits_assignment)
    pickle.dump(solving_time, open("" + layer_key + "_solving_time.dat", "wb"))
    pickle.dump(bits_assignment, open("" + layer_key + "_bits_assignment.dat", "wb"))
    pickle.dump(0, open("" + layer_key, "wb"))
    return

def test_func():
    try:
        fd = inotify_simple.INotify()
        wd = fd.add_watch("", inotify_simple.flags.MODIFY | inotify_simple.flags.CREATE )
        while True:
            print("thread just looping")
            for event in fd.read():
                if event.mask & (inotify_simple.flags.MODIFY | inotify_simple.flags.CREATE):
                    print("found file!!!")
                    on_new_file(event)
    finally:
        try:
            print("Closed instance")
            fd.close()
        except Exception as e:
            print("exception")
            logger.exception("An error occurred while closing the INotify instance.")

def real_func():
    directory_to_watch = ""
    watcher_thread = threading.Thread(target=test_func)
    watcher_thread.start()


# bits_set = pickle.load(open("bits_set.dat", "rb"))
# var_matrix = pickle.load(open("var_matrix.dat", "rb"))
# comm_matrix = pickle.load(open("comm_matrix.dat", "rb"))
# cost_model = pickle.load(open("cost_model.dat", "rb"))
# coe_lambda = pickle.load(open("coe_lambda.dat", "rb"))
# world_size = pickle.load(open("world_size.dat", "rb"))
# normal_mode = pickle.load(open("normal_mode.dat", "rb"))

# get_solution(bits_set, var_matrix, comm_matrix, cost_model, coe_lambda, world_size, normal_mode)

#real_func()

shutil.rmtree("")
shutil.rmtree("")
shutil.rmtree("")

os.makedirs("", exist_ok=True)
os.makedirs("", exist_ok=True)
os.makedirs("", exist_ok=True)

while True:
    for file in os.listdir(os.fsencode("")):
        filename = os.fsdecode(file)
        first_time = pickle.load(open("" + filename, "rb"))
        os.remove("" + filename)
        worker_thread = threading.Thread(target=lambda: on_new_file(filename, first_time))
        worker_thread.start()
    time.sleep(0.1)

