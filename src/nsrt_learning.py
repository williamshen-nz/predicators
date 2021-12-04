"""Algorithms for learning the various components of NSRT objects.
"""

import functools
import itertools
from typing import Set, Tuple, List, Sequence, FrozenSet
from predicators.src.structs import Dataset, STRIPSOperator, NSRT, \
    GroundAtom, LiftedAtom, Variable, Predicate, ObjToVarSub, \
    StateActionTrajectory, Segment, Partition, Object, GroundAtomTrajectory, \
    DefaultOption, ParameterizedOption
from predicators.src import utils
from predicators.src.settings import CFG
from predicators.src.sampler_learning import learn_samplers
from predicators.src.option_learning import create_option_learner


def learn_nsrts_from_data(dataset: Dataset, predicates: Set[Predicate],
                          do_sampler_learning: bool) -> Set[NSRT]:
    """Learn NSRTs from the given dataset of transitions.
    States are abstracted using the given set of predicates.
    """
    print(f"\nLearning NSRTs on {len(dataset)} trajectories...")

    # Apply predicates to dataset.
    ground_atom_dataset = utils.create_ground_atom_dataset(dataset, predicates)

    # Segment transitions based on changes in predicates.
    segments = [seg for traj in ground_atom_dataset
                for seg in segment_trajectory(traj)]

    # Learn strips operators.
    strips_ops, partitions = learn_strips_operators(segments)
    assert len(strips_ops) == len(partitions)

    # Learn option specs, or if known, just look them up. The order of
    # the options corresponds to the strips_ops. Each spec is a
    # (ParameterizedOption, Sequence[Variable]) tuple with the latter
    # holding the option_vars. After learning the specs, update the
    # segments to include which option is being executed within each
    # segment, so that sampler learning can utilize this.
    option_learner = create_option_learner()
    option_specs = option_learner.learn_option_specs(strips_ops, partitions)
    assert len(option_specs) == len(strips_ops)
    for partition, spec in zip(partitions, option_specs):
        for (segment, _) in partition:
            # Modifies segment in-place.
            option_learner.update_segment_from_option_spec(segment, spec)

    # Learn samplers.
    # The order of the samplers also corresponds to strips_ops.
    samplers = learn_samplers(strips_ops, partitions, option_specs,
                              do_sampler_learning)
    assert len(samplers) == len(strips_ops)

    # Create final NSRTs.
    nsrts = []
    for op, option_spec, sampler in zip(strips_ops, option_specs, samplers):
        param_option, option_vars = option_spec
        nsrt = op.make_nsrt(param_option, option_vars, sampler)
        nsrts.append(nsrt)

    print("\nLearned NSRTs:")
    for nsrt in sorted(nsrts):
        print(nsrt)
    print()

    return set(nsrts)


def segment_trajectory(trajectory: GroundAtomTrajectory) -> List[Segment]:
    """Segment a ground atom trajectory according to abstract state changes.

    If options are available, also use them to segment.
    """
    segments = []
    states, actions, all_atoms = trajectory
    assert len(states) == len(actions) + 1 == len(all_atoms)
    current_segment_traj : StateActionTrajectory = ([], [])
    for t in range(len(actions)):
        current_segment_traj[0].append(states[t])
        current_segment_traj[1].append(actions[t])
        switch = all_atoms[t] != all_atoms[t+1]
        # Segment based on option specs if we are assuming that options are
        # known. If we do not do this, it can lead to a bug where an option
        # has object arguments that do not appear in the strips operator
        # parameters. Note also that we are segmenting based on option specs,
        # rather than option changes. This distinction is subtle but important.
        # For example, in Cover, there is just one parameterized option, which
        # is PickPlace() with no object arguments. If we segmented based on
        # option changes, then segmentation would break up trajectories into
        # picks and places. Then, when operator learning, it would appear
        # that no predicates are necessary to distinguish between picking
        # and placing, since the option changes and segmentation have already
        # made the distinction. But we want operator learning to use predicates
        # like Holding, Handempty, etc., because when doing symbolic planning,
        # we only have predicates, and not the continuous parameters that would
        # be used to distinguish between a PickPlace that is a pick vs a place.
        if actions[t].has_option() and t < len(actions) - 1:
            # We don't care about the last option in the traj because there's
            # no next option after it, so we could never possibly want to
            # segment due to that option. But we still need to check atoms here,
            # because the length of all_atoms is one longer.
            option_t = actions[t].get_option()
            option_t1 = actions[t+1].get_option()
            option_t_spec = (option_t.parent, option_t.objects)
            option_t1_spec = (option_t1.parent, option_t1.objects)
            if option_t_spec != option_t1_spec:
                switch = True
        if switch:
            # Include the final state as the end of this segment.
            current_segment_traj[0].append(states[t+1])
            if actions[t].has_option():
                segment = Segment(current_segment_traj, all_atoms[t].copy(),
                                  all_atoms[t+1].copy(),
                                  actions[t].get_option())
            else:
                # If option learning, include the default option here; replaced
                # during option learning.
                segment = Segment(current_segment_traj,
                                  all_atoms[t].copy(), all_atoms[t+1].copy())
            segments.append(segment)
            current_segment_traj = ([], [])
    # Don't include the last current segment because it didn't result in
    # an abstract state change. (E.g., the option may not be terminating.)
    return segments


def learn_strips_operators(segments: Sequence[Segment], verbose: bool = True,
        ) -> Tuple[List[STRIPSOperator], List[Partition]]:
    """Learn operators given the segmented transitions.
    """
    # Partition the segments according to common effects.
    params: List[Sequence[Variable]] = []
    parameterized_options: List[ParameterizedOption] = []
    option_vars: List[Tuple[Variable, ...]] = []
    add_effects: List[Set[LiftedAtom]] = []
    delete_effects: List[Set[LiftedAtom]] = []
    partitions: List[Partition] = []
    for segment in segments:
        if segment.has_option():
            segment_option = segment.get_option()
            segment_param_option = segment_option.parent
            segment_option_objs = tuple(segment_option.objects)
        else:
            segment_param_option = DefaultOption.parent
            segment_option_objs = tuple()
        for i in range(len(partitions)):
            # Try to unify this transition with existing effects.
            # Note that both add and delete effects must unify,
            # and also the objects that are arguments to the options.
            part_param_option = parameterized_options[i]
            part_option_vars = option_vars[i]
            part_add_effects = add_effects[i]
            part_delete_effects = delete_effects[i]
            suc, sub = unify_effects_and_options(
                frozenset(segment.add_effects),
                frozenset(part_add_effects),
                frozenset(segment.delete_effects),
                frozenset(part_delete_effects),
                segment_param_option,
                part_param_option,
                segment_option_objs,
                part_option_vars)
            if suc:
                # Add to this partition.
                assert set(sub.values()) == set(params[i])
                partitions[i].add((segment, sub))
                break
        # Otherwise, create a new group.
        else:
            # Get new lifted effects.
            objects = {o for atom in segment.add_effects |
                       segment.delete_effects for o in atom.objects} | \
                      set(segment_option_objs)
            objects_lst = sorted(objects)
            variables = [Variable(f"?x{i}", o.type)
                         for i, o in enumerate(objects_lst)]
            sub = dict(zip(objects_lst, variables))
            params.append(variables)
            parameterized_options.append(segment_param_option)
            option_vars.append(tuple(sub[o] for o in segment_option_objs))
            add_effects.append({atom.lift(sub) for atom
                                in segment.add_effects})
            delete_effects.append({atom.lift(sub) for atom
                                   in segment.delete_effects})
            new_partition = Partition([(segment, sub)])
            partitions.append(new_partition)

    # We don't need option_vars anymore; we'll recover them later when we call
    # `learn_option_specs`. The only reason to include them here is to make sure
    # that params include the option_vars when options are available.
    del option_vars

    assert len(params) == len(add_effects) == \
           len(delete_effects) == len(partitions)

    # The effects are now stored in the partitions.
    del add_effects, delete_effects

    # Prune partitions with not enough data.
    kept_idxs = []
    for idx, partition in enumerate(partitions):
        if len(partition) >= CFG.min_data_for_nsrt:
            kept_idxs.append(idx)
    params = [params[i] for i in kept_idxs]
    partitions = [partitions[i] for i in kept_idxs]

    # Add ForallNot predicates to partitions.
    for i, p in enumerate(partitions):
        _add_forallnots(p)

    # Learn preconditions.
    preconds = [_learn_preconditions(p) for p in partitions]

    # Finalize the operators.
    ops = []
    for i in range(len(params)):
        name = f"Op{i}"
        partition = partitions[i]
        op = STRIPSOperator(name, params[i], preconds[i],
                            partition.add_effects,
                            partition.delete_effects)
        if verbose:
            print("Learned STRIPSOperator:")
            print(op)
        ops.append(op)

    return ops, partitions


def _add_forallnots(partition: Partition) -> None:
    # Create all possible ForallNot predicates using the predicates and
    # variables in the partition.
    predicates = set()
    variables = set()
    for i, (segment, sub) in enumerate(partition):
        segment_predicates = {atom.predicate \
            for atom in segment.init_atoms | segment.final_atoms}
        predicates.update(segment_predicates)
        segment_variables = set(sub.values())
        if i == 0:
            variables = segment_variables
        assert variables == segment_variables
    forallnot_predicates = set()
    for predicate in predicates:
        for num_free_indices in range(predicate.arity):
            for free_indices in itertools.combinations(range(predicate.arity),
                                                       num_free_indices):
                set_free_indices = set(free_indices)
                forallnot_predicate = utils.create_forall_not_predicate(
                    predicate, set_free_indices)
                forallnot_predicates.add((predicate, forallnot_predicate))
    # Abstract the segments in the partition. For the free variables,
    # only instantiate the predicate for objects that appear in the
    # segment's ObjToVarSub. This is an optimization that saves us
    # from needing to quantify over all objects in the state. Also note:
    # this optimization is the reason that we add the forallnots here
    # rather than as a preprocessing step to the whole learning pipeline.
    for segment, sub in partition:
        # Initially assume that all forallnot predicates appear in the
        # segment; remove as we find counterexamples. This process is
        # why we're specifically doing "forallnot" as opposed to "forall".
        segment_forall_not_atoms = {}
        for predicate, fan_predicate in forallnot_predicates:
            # Get all possible groundings using the relevant objects.
            segment_objects = set(sub.keys())
            for choice in utils.get_object_combinations(segment_objects,
                                                        fan_predicate.types):
                forallnot_atom_id = (predicate, frozenset(enumerate(choice)))
                forall_not_atom = GroundAtom(fan_predicate, choice)
                segment_forall_not_atoms[forallnot_atom_id] = forall_not_atom
        new_init_atoms = segment_forall_not_atoms.copy()
        new_final_atoms = segment_forall_not_atoms.copy()
        # Find counterexamples and remove forallnots accordingly.
        for atom_set, new_atoms in [(segment.init_atoms, new_init_atoms),
                                    (segment.final_atoms, new_final_atoms)]:
            for atom in atom_set:
                # Each atom can create a number of counterexamples.
                # For example, IsBlock(block0) is a counterexample for
                # FORALL-NOT-0-IsBlock(). More interestingly,
                # Covers(block0, target0) is a counterexample for
                # FORALL-NOT-0-Covers(target0), FORALL-NOT-1-Covers(block0),
                # and FORALL-NOT-0,1-Covers(). In general, every
                # subset of the atom arguments leads to one counterexample.
                pred = atom.predicate
                for k in range(pred.arity+1):
                    for idxs in itertools.combinations(range(pred.arity), k):
                        forallnot_atom_id = (pred,
                            frozenset((i, atom.objects[i]) for i in idxs))
                        if forallnot_atom_id in new_atoms:
                            # Useful for debugging:
                            # print("Removing", new_atoms[forallnot_atom_id])
                            # print("from atom set", atom_set)
                            # print("because of ", atom)
                            del new_atoms[forallnot_atom_id]
            # Update the atom set in place.
            # Useful for debugging:
            # print("Adding", set(new_atoms.values()))
            # print("to atom set", atom_set)
            # print("with relevant objects", segment_objects)
            atom_set.update(new_atoms.values())


def  _learn_preconditions(partition: Partition) -> Set[LiftedAtom]:
    for i, (segment, sub) in enumerate(partition):
        atoms = segment.init_atoms
        objects = set(sub.keys())
        atoms = {atom for atom in atoms if
                 all(o in objects for o in atom.objects)}
        lifted_atoms = {atom.lift(sub) for atom in atoms}
        if i == 0:
            variables = sorted(set(sub.values()))
        else:
            assert variables == sorted(set(sub.values()))
        if i == 0:
            preconditions = lifted_atoms
        else:
            preconditions &= lifted_atoms
    return preconditions


@functools.lru_cache(maxsize=None)
def unify_effects_and_options(
        ground_add_effects: FrozenSet[GroundAtom],
        lifted_add_effects: FrozenSet[LiftedAtom],
        ground_delete_effects: FrozenSet[GroundAtom],
        lifted_delete_effects: FrozenSet[LiftedAtom],
        ground_param_option: ParameterizedOption,
        lifted_param_option: ParameterizedOption,
        ground_option_args: Tuple[Object, ...],
        lifted_option_args: Tuple[Variable, ...]
) -> Tuple[bool, ObjToVarSub]:
    """Wrapper around utils.unify() that handles option arguments, add effects,
    and delete effects. Changes predicate names so that all are treated
    differently by utils.unify().
    """
    # Can't unify if the parameterized options are different.
    # Note, of course, we could directly check this in the loop above. But we
    # want to keep all the unification logic in one place, even if it's trivial
    # in this case.
    if ground_param_option != lifted_param_option:
        return False, {}
    ground_opt_arg_pred = Predicate("OPT-ARGS",
                                    [a.type for a in ground_option_args],
                                    _classifier=lambda s, o: False)  # dummy
    f_ground_option_args = frozenset({GroundAtom(ground_opt_arg_pred,
                                                 ground_option_args)})
    new_ground_add_effects = utils.wrap_atom_predicates_ground(
        ground_add_effects, "ADD-")
    f_new_ground_add_effects = frozenset(new_ground_add_effects)
    new_ground_delete_effects = utils.wrap_atom_predicates_ground(
        ground_delete_effects, "DEL-")
    f_new_ground_delete_effects = frozenset(new_ground_delete_effects)

    lifted_opt_arg_pred = Predicate("OPT-ARGS",
                                    [a.type for a in lifted_option_args],
                                    _classifier=lambda s, o: False)  # dummy
    f_lifted_option_args = frozenset({LiftedAtom(lifted_opt_arg_pred,
                                                 lifted_option_args)})
    new_lifted_add_effects = utils.wrap_atom_predicates_lifted(
        lifted_add_effects, "ADD-")
    f_new_lifted_add_effects = frozenset(new_lifted_add_effects)
    new_lifted_delete_effects = utils.wrap_atom_predicates_lifted(
        lifted_delete_effects, "DEL-")
    f_new_lifted_delete_effects = frozenset(new_lifted_delete_effects)
    return utils.unify(
        f_ground_option_args | f_new_ground_add_effects | \
            f_new_ground_delete_effects,
        f_lifted_option_args | f_new_lifted_add_effects | \
            f_new_lifted_delete_effects)
