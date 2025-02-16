"""Create offline datasets for training, given a set of training tasks for an
environment."""

from typing import List
from predicators.src.envs import BaseEnv
from predicators.src.structs import Dataset, Task
from predicators.src.datasets.demo_only import create_demo_data
from predicators.src.datasets.demo_replay import create_demo_replay_data
from predicators.src.datasets.ground_atom_data import create_ground_atom_data
from predicators.src.settings import CFG
from predicators.src import utils


def create_dataset(env: BaseEnv, train_tasks: List[Task]) -> Dataset:
    """Create offline datasets for training, given a set of training tasks for
    an environment."""
    if CFG.offline_data_method == "demo":
        return create_demo_data(env, train_tasks)
    if CFG.offline_data_method == "demo+replay":
        return create_demo_replay_data(env, train_tasks)
    if CFG.offline_data_method == "demo+nonoptimalreplay":
        return create_demo_replay_data(env, train_tasks, nonoptimal_only=True)
    if CFG.offline_data_method == "demo+ground_atoms":
        base_dataset = create_demo_data(env, train_tasks)
        _, excluded_preds = utils.parse_config_excluded_predicates(env)
        n = int(CFG.teacher_dataset_num_examples)
        assert n >= 1, "Must have at least 1 example of each predicate"
        return create_ground_atom_data(env, base_dataset, excluded_preds, n)
    if CFG.offline_data_method == "empty":
        return Dataset([])
    raise NotImplementedError("Unrecognized dataset method.")
