"""An approach that learns predicates from a teacher."""

from typing import Set, List, Optional, Tuple, Callable, Sequence, Dict
import dill as pkl
import numpy as np
from gym.spaces import Box
from predicators.src import utils
from predicators.src.approaches import NSRTLearningApproach, \
    ApproachTimeout, ApproachFailure
from predicators.src.structs import State, Predicate, ParameterizedOption, \
    Type, Task, Dataset, GroundAtom, LowLevelTrajectory, InteractionRequest, \
    InteractionResult, Action
from predicators.src.torch_models import LearnedPredicateClassifier, \
    MLPClassifier
from predicators.src.teacher import GroundAtomsHoldQuery, \
    GroundAtomsHoldResponse, Query
from predicators.src.settings import CFG


class InteractiveLearningApproach(NSRTLearningApproach):
    """An approach that learns predicates from a teacher."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # Track score of best atom seen so far.
        self._best_score = 0.0
        # Initialize things that will be set correctly in offline learning.
        self._dataset = Dataset([], [])
        self._predicates_to_learn: Set[Predicate] = set()
        self._online_learning_cycle = 0

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._initial_predicates | self._predicates_to_learn

    ######################## Semi-supervised learning #########################

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # First, go through the dataset's annotations and figure out the
        # set of predicates to learn. Note that their classifiers were
        # stripped away during the creation of the annotations.
        for ground_atom_sets in dataset.annotations:
            for ground_atom_set in ground_atom_sets:
                for atom in ground_atom_set:
                    assert atom.predicate not in self._initial_predicates
                    self._predicates_to_learn.add(atom.predicate)
        # Next, convert the dataset with positive annotations only into a
        # dataset with positive and unlabeled annotations.
        new_annotations = []
        for traj, ground_atom_sets in zip(dataset.trajectories,
                                          dataset.annotations):
            new_traj_annotation = []
            # Get all possible ground atoms given the objects in traj.
            possible = set(
                utils.all_possible_ground_atoms(traj.states[0],
                                                self._predicates_to_learn))
            for positives in ground_atom_sets:
                unlabeled = possible - positives
                new_traj_annotation.append({
                    "positive": positives,
                    "unlabeled": unlabeled,
                    "negative": set(),
                })
            new_annotations.append(new_traj_annotation)
        self._dataset = Dataset(dataset.trajectories, new_annotations)
        # Learn predicates and NSRTs.
        self._relearn_predicates_and_nsrts(online_learning_cycle=None)

    def load(self, online_learning_cycle: Optional[int]) -> None:
        super().load(online_learning_cycle)
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_{online_learning_cycle}.DATA", "rb") as f:
            save_dict = pkl.load(f)
        self._dataset = save_dict["dataset"]
        self._predicates_to_learn = save_dict["predicates_to_learn"]
        self._best_score = save_dict["best_score"]

    def _relearn_predicates_and_nsrts(
            self, online_learning_cycle: Optional[int]) -> None:
        """Learns predicates and NSRTs in a semi-supervised fashion."""
        print("\nRelearning predicates and NSRTs...")
        # Learn predicates
        for pred in self._predicates_to_learn:
            input_examples = []
            output_examples = []
            for (traj, traj_annotations) in zip(self._dataset.trajectories,
                                                self._dataset.annotations):
                assert len(traj.states) == len(traj_annotations)
                for (state, state_annotation) in zip(traj.states,
                                                     traj_annotations):
                    # Here we make the (wrong in general!) assumption that
                    # unlabeled ground atoms are negative. In the future, we
                    # may want to modify this, e.g., downweight or remove
                    # the unlabeled examples once we collect enough negatives.
                    for label, target_class in [("positive", 1),
                                                ("unlabeled", 0),
                                                ("negative", 0)]:
                        for atom in state_annotation[label]:
                            if not atom.predicate == pred:
                                continue
                            x = state.vec(atom.objects)
                            input_examples.append(x)
                            output_examples.append(target_class)
            num_positives = sum(y == 1 for y in output_examples)
            num_negatives = sum(y == 0 for y in output_examples)
            assert num_positives + num_negatives == len(output_examples)
            print(f"Generated {num_positives} positive and "
                  f"{num_negatives} negative examples for "
                  f"predicate {pred}")

            # Train MLP
            X = np.array(input_examples)
            Y = np.array(output_examples)
            model = MLPClassifier(X.shape[1],
                                  CFG.predicate_mlp_classifier_max_itr)
            model.fit(X, Y)

            # Construct classifier function, create new Predicate, and save it
            classifier = LearnedPredicateClassifier(model).classifier
            new_pred = Predicate(pred.name, pred.types, classifier)
            self._predicates_to_learn = \
                (self._predicates_to_learn - {pred}) | {new_pred}

        # Learn NSRTs via superclass
        self._learn_nsrts(self._dataset.trajectories, online_learning_cycle)

        # Save the things we need other than the NSRTs, which were already
        # saved in the above call to self._learn_nsrts()
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_{online_learning_cycle}.DATA", "wb") as f:
            pkl.dump(
                {
                    "dataset": self._dataset,
                    "predicates_to_learn": self._predicates_to_learn,
                    "best_score": self._best_score,
                }, f)

    ########################### Active learning ###############################

    def get_interaction_requests(self) -> List[InteractionRequest]:
        # We will create a single interaction request.
        # Determine the train task that we will be using.
        train_task_idx = self._select_interaction_train_task_idx()
        # Determine the action policy and termination function.
        act_policy, termination_function = \
            self._create_interaction_action_strategy(train_task_idx)
        # Determine the query policy.
        query_policy = self._create_interaction_query_policy(train_task_idx)
        return [
            InteractionRequest(train_task_idx, act_policy, query_policy,
                               termination_function)
        ]

    def _score_atom_set(self, atom_set: Set[GroundAtom],
                        state: State) -> float:
        """Score an atom set based on how much we would like to know the values
        of all the atoms in the set in the given state.

        Higher scores are better.
        """
        del state  # not currently used, but will be by future score functions
        if CFG.interactive_score_function == "frequency":
            return self._score_atom_set_frequency(atom_set)
        raise NotImplementedError("Unrecognized interactive_score_function:"
                                  f" {CFG.interactive_score_function}.")

    def _select_interaction_train_task_idx(self) -> int:
        # At the moment, we only have one way to select a train task idx:
        # choose one uniformly at random. In the future, we may want to
        # try other strategies. But one nice thing about random selection
        # is that we're not making a hard commitment to the agent having
        # control over which train task it gets to use.
        return self._rng.choice(len(self._train_tasks))

    def _create_interaction_action_strategy(
        self, train_task_idx: int
    ) -> Tuple[Callable[[State], Action], Callable[[State], bool]]:
        """Returns an action policy and a termination function."""
        if CFG.interactive_action_strategy == "glib":
            return self._create_glib_interaction_strategy(train_task_idx)
        raise NotImplementedError("Unrecognized interactive_action_strategy:"
                                  f" {CFG.interactive_action_strategy}")

    def _create_interaction_query_policy(
            self, train_task_idx: int) -> Callable[[State], Optional[Query]]:
        """Returns a query policy."""
        del train_task_idx  # unused right now, but future policies may use
        if CFG.interactive_query_policy == "strict_best_seen":
            return self._create_best_seen_query_policy()
        raise NotImplementedError("Unrecognized interactive_query_policy:"
                                  f" {CFG.interactive_query_policy}")

    def _create_glib_interaction_strategy(
        self, train_task_idx: int
    ) -> Tuple[Callable[[State], Action], Callable[[State], bool]]:
        """Find the most interesting reachable ground goal and plan to it."""
        init = self._train_tasks[train_task_idx].init
        # Detect and filter out static predicates.
        static_preds = utils.get_static_preds(self._nsrts,
                                              self._predicates_to_learn)
        preds = self._predicates_to_learn - static_preds
        # Sample possible goals to plan toward.
        ground_atom_universe = utils.all_possible_ground_atoms(init, preds)
        possible_goals = utils.sample_subsets(
            ground_atom_universe,
            num_samples=CFG.interactive_num_babbles,
            min_set_size=1,
            max_set_size=CFG.interactive_max_num_atoms_babbled,
            rng=self._rng)
        # Sort the possible goals based on how interesting they are.
        # Note: we're using _score_atom_set_frequency here instead of
        # _score_atom_set because _score_atom_set in general could depend
        # on the current state. While babbling goals, we don't have any
        # current state because we don't know what the state will be if and
        # when we get to the goal.
        goal_list = sorted(possible_goals,
                           key=self._score_atom_set_frequency,
                           reverse=True)  # largest to smallest
        task_list = [Task(init, goal) for goal in goal_list]
        task, act_policy = self._find_first_solvable(task_list)
        assert task.init is init

        def _termination_function(s: State) -> bool:
            # Stop the episode if we reach the goal that we babbled.
            return all(goal_atom.holds(s) for goal_atom in task.goal)

        return act_policy, _termination_function

    def _create_best_seen_query_policy(
            self) -> Callable[[State], Optional[Query]]:
        """Only query if the atom has the best score seen so far."""

        def _query_policy(s: State) -> Optional[GroundAtomsHoldQuery]:
            # Decide whether to ask about each possible atom.
            ground_atoms = utils.all_possible_ground_atoms(
                s, self._predicates_to_learn)
            atoms_to_query = set()
            for atom in ground_atoms:
                score = self._score_atom_set({atom}, s)
                # Ask about this atom if it is the best seen so far.
                if score > self._best_score:
                    atoms_to_query.add(atom)
                    self._best_score = score
            return GroundAtomsHoldQuery(atoms_to_query)

        return _query_policy

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        assert len(results) == 1
        result = results[0]
        assert len(result.states) == len(result.responses) == \
            len(result.actions) + 1
        new_annotations = []
        for state, response in zip(result.states, result.responses):
            assert isinstance(response, GroundAtomsHoldResponse)
            state_annotation: Dict[str, Set[GroundAtom]] = {
                "positive": set(),
                "negative": set(),
                "unlabeled": set()
            }
            for query_atom, atom_holds in response.holds.items():
                label = "positive" if atom_holds else "negative"
                state_annotation[label].add(query_atom)
            new_annotations.append(state_annotation)
        traj = LowLevelTrajectory(result.states, result.actions)
        self._dataset.append(traj, new_annotations)
        self._relearn_predicates_and_nsrts(
            online_learning_cycle=self._online_learning_cycle)
        self._online_learning_cycle += 1

    def _find_first_solvable(
            self,
            task_list: List[Task]) -> Tuple[Task, Callable[[State], Action]]:
        for task in task_list:
            try:
                print("Solving for policy...")
                policy = self.solve(task, timeout=CFG.timeout)
                return task, policy
            except (ApproachTimeout, ApproachFailure) \
                    as e:  # pragma: no cover
                print(f"Approach failed to solve with error: {e}")
                continue
        raise ApproachFailure("Failed to sample a task that approach "
                              "can solve.")  # pragma: no cover

    def _score_atom_set_frequency(self, atom_set: Set[GroundAtom]) -> float:
        """Score an atom set as inversely proportional to the number of
        examples seen during training."""
        count = 1  # Avoid division by 0
        for ground_atom_traj in self._dataset.annotations:
            for ground_atom_set in ground_atom_traj:
                count += 1 if atom_set.issubset(ground_atom_set) else 0
        return 1.0 / count
