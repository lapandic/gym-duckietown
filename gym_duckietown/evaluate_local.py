"""
Evaluate local is used to compute the AI-DO scores of the learned model
in simulation on your local computer.
It uses:
    - The learned model.
    - The created evaluation data log.
It produces:
    - A set of metrics of performance over time as well as averaged over time.
"""

import os
import h5py
import numpy as np

PATH = 'eval_data.h5'
TAGS = ['imgs', 'reward', 'actions', 'xyz',
        'angle', 'velocity', 'ref-xyz', 'ref-dir', 'daytime', 'ID']
INITIAL_SIZE = 10**5  # Number of evaluation steps


def create_eval_log():
    # run specific instances
    return 0

def extract_data():
    tags = {
                'Images': [(initial_size, 120, 160, 3), 'i1'],
                 'Reward': [(initial_size, 1), 'float'],
                 'Output': [(initial_size, 2), 'float'],
                 'Position': [(initial_size, 3), 'float'],
                 'Angle': [(initial_size,), 'float'],
                 'Velocity': [(initial_size,), 'float'],
                 'Ref-Position': [(initial_size, 3), 'float']}

     if not os.path.exists(path):
         # f = h5py.File(path, "w")
         # f.close()
         with h5py.File(path, "r") as f:
             print("tags", tags)
             for tag, attribute in tags.items():
                 print("tag", tag)
                 print("attribute", attribute)
                 maxshape_value = list(attribute[0])
                 maxshape_value[0] = None
                 print("Max shape:", tuple(maxshape_value))
                 f.create_dataset(tag, shape=attribute[0],
                                  dtype=attribute[1],
                                  maxshape=tuple(maxshape_value))
    return 0


def evaluate_logs():
    return 0




    return 0

def p_lf(vel, ref_direction):
    """
    Performance: - Lane following (LF) and
                 - lane following with dynamic vehicles (LFV)
    Rules specified in: http://docs.duckietown.org/AIDO/out/performance.html

    General principle: High velocity is encouraged parallel to the the road.

    The velocity is aligned to the road through a dot product.
    Note that the sign is negative since objectives are minimized.
    vel: numpy array of shape (n_time_steps, 3), dtype float
    ref_direction: numpy array of shape (n_time_steps, 3), dtype float
    """
    #TODO: How to extract ref_direction?
    #TODO: logs need ref pos. and ref dir.
    assert(vel.shape == ref_direction.shape)
    return -np.sum(np.dot(vel, ref_direction.T))

def p_navv(data_log):
    """
    Performance: - Navigation with dynamic vehicles (NAVV)
    Rules specified in: http://docs.duckietown.org/AIDO/out/performance.html
    Accumulates cost as long as destination is not reached.
    """
    #TODO: to be filled in - function nav_is_active missing
    mask = nav_is_active(data_log)

    return np.sum(mask)

def rule_stay_in_lane():
    return 0


if __name__ ==  "__main__":
    """
    - Load model: user provides path to model
    - Run model in simulator for NSTEPS
    - Save statistics using datalogger
    - Calculate scores from this
    """

    log_name = 'eval_1'
    model = load_model()
    create_eval_log(model, log_name)
    eval_log = extract_data(log_name)
    perf_lf = p_lf(eval_log)
