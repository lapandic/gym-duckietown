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

CAMERA_HEIGHT = 120
CAMERA_WIDTH = 160

def create_eval_log():
    # run specific instances
    return 0


class Evaluator():

    def __init__(self, path):
        self.initial_size = 0
        self.tags = {
         'image': [[self.initial_size, CAMERA_HEIGHT, CAMERA_WIDTH, 3], np.uint8],
         'reward': [[self.initial_size, 1], np.float32],
         'actions': [[self.initial_size, 2], np.float32],
         'position': [[self.initial_size, 3], np.float32],
         'dir': [[self.initial_size], np.float32],
         'speed': [[self.initial_size], np.float32],
         'ref_position': [[self.initial_size, 3], np.float32],
         'ref_dir': [[self.initial_size], np.float32],
         # 'time_stamp': [[self.initial_size], np.string_]
        }
        self.pos_traj = 'position'
        self.ref_traj = 'ref_position'
        self.dir = 'dir'
        self.ref_dir = 'ref_dir'
        self.path = path
        if not os.path.exists(self.path):
             FileNotFoundError()

    # def extract_data(self):
    #
    #          # f = h5py.File(path, "w")
    #          # f.close()
    #      # with h5py.File(path, "r") as f:
    #
    #          self.pos_traj = 'position'
    #          self.ref_traj = 'ref_position'
    #          self.dir = 'dir'
    #          self.ref_dir = 'ref_dir'

    def evaluate_logs(self):
        """
        Runs everything at once to get log evaluation summary
        """

        #TODO: create one big stream over the evaluation log
        #TODO: give relevant data to evaluation functions
        #TODO: aggregate results, compute statistics

        with h5py.File(self.path, "r") as f:
            rule_infractions = np.zeros((f[self.pos_traj].shape[0],))
            for i in range(f[self.pos_traj].shape[0]):
                pos = f[self.pos_traj][i]
                ref_pos = f[self.ref_traj][i]
                dir = f[self.dir][i]
                ref_dir = f[self.ref_dir][i]
                rule_infractions[i] = self.rule_stay_in_lane(pos, ref_pos)

        print(rule_infractions)
        return 0

    def evaluation_statistics(self):
        """

        :return:
        """
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

    def rule_stay_in_lane(self, pos, ref_pos):
        beta = 0.1
        alpha = 5
        d_safe = 0.05
        d_max = 0.4

        diff = pos - ref_pos
        if diff < d_safe:
            penalty = 0
        elif d_safe <= diff <= d_max:
            penalty = beta * diff**2
        else:
            penalty = alpha
        yield penalty


# if __name__ ==  "__main__":
#     """
#     - Load model: user provides path to model
#     - Run model in simulator for NSTEPS
#     - Save statistics using datalogger
#     - Calculate scores from this
#     """
#
#     log_name = 'eval_1'
#     model = load_model()
#     create_eval_log(model, log_name)
#     eval_log = extract_data(log_name)
#     perf_lf = p_lf(eval_log)
