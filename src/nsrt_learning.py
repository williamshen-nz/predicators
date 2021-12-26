"""Algorithms for learning the various components of NSRT objects.
"""

import functools
from typing import Set, Tuple, List, Sequence, FrozenSet
from predicators.src.structs import Dataset, STRIPSOperator, NSRT, \
    GroundAtom, LiftedAtom, Variable, Predicate, ObjToVarSub, \
    LowLevelTrajectory, Segment, Partition, Object, GroundAtomTrajectory, \
    DummyOption, ParameterizedOption, State, Action
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
    segmented_trajectories = [segment_trajectory(traj)
                              for traj in ground_atom_dataset]
    segments = [seg for segmented_traj in segmented_trajectories
                for seg in segmented_traj]

    # Learn strips operators.
    strips_ops, partitions, parameterized_options, option_vars = \
        learn_strips_operators(segments, verbose=CFG.do_option_learning)
    assert len(strips_ops) == len(partitions) == len(parameterized_options) == \
        len(option_vars)

    # Try to prune the strips operator effects and add side predicates.
    demo_trajectories = [segment_trajectory(traj)
                         for traj in ground_atom_dataset if traj[0].is_demo]
    goals = [traj[0].goal for traj in ground_atom_dataset if traj[0].is_demo]
    prune_operator_effects(strips_ops, demo_trajectories, goals,
                           parameterized_options, option_vars)
    # prune_operator_effects(strips_ops, segmented_trajectories)

    # Re-partition the data. Note now that each transition could end up
    # in multiple partitions.
    new_partitions = [[] for _ in partitions]
    for segment in segments:
        # TODO make less stupid.
        if segment.has_option():
            segment_option = segment.get_option()
            segment_param_option = segment_option.parent
            segment_option_objs = tuple(segment_option.objects)
        else:
            segment_param_option = DummyOption.parent
            segment_option_objs = tuple()
        for i in range(len(partitions)):
            if segment_param_option != parameterized_options[i]:
                continue
            op = strips_ops[i]
            assert len(option_vars[i]) == len(segment_option_objs)
            partial_inv_sub = dict(zip(option_vars[i], segment_option_objs))
            for ground_op in utils.all_ground_operators_given_partial(op,
                set(segment.states[0]), partial_inv_sub):
                if not ground_op.preconditions.issubset(segment.init_atoms):
                    continue
                atoms = utils.apply_operator(ground_op, segment.init_atoms)
                
                # !!!!!
                # if atoms.issuperset(segment.final_atoms):
                if atoms == segment.final_atoms:

                    full_inv_sub = dict(zip(op.parameters, ground_op.objects))
                    assert all(partial_inv_sub[k] == full_inv_sub[k] for k in partial_inv_sub)

                    # TODO: ??!
                    if len(set(full_inv_sub.values())) != len(full_inv_sub):
                        continue
                    full_sub = {v: k for k, v in full_inv_sub.items()}

                    new_partitions[i].append((segment, full_sub))

    for i in range(len(new_partitions)-1, -1, -1):
        if not new_partitions[i]:
            del strips_ops[i]
            del new_partitions[i]
    partitions = [Partition(members) for members in new_partitions
                  if members]

    # TODO: remove redundant operators.

    # Learn option specs, or if known, just look them up. The order of
    # the options corresponds to the strips_ops. Each spec is a
    # (ParameterizedOption, Sequence[Variable]) tuple with the latter
    # holding the option_vars. After learning the specs, update the
    # segments to include which option is being executed within each
    # segment, so that sampler learning can utilize this.
    option_learner = create_option_learner()
    option_specs = option_learner.learn_option_specs(strips_ops, partitions)
    assert len(option_specs) == len(strips_ops)
    # Seed the new parameterized option parameter spaces.
    for parameterized_option, _ in option_specs:
        parameterized_option.params_space.seed(CFG.seed)
    # Update the segments to include which option is being executed.
    for partition, spec in zip(partitions, option_specs):
        for (segment, _) in partition:
            # Modifies segment in-place.
            option_learner.update_segment_from_option_spec(segment, spec)

    # For the impatient, print out the STRIPSOperators with their option specs.
    print("\nLearned operators with option specs:")
    for strips_op, (option, option_vars) in zip(strips_ops, option_specs):
        print(strips_op)
        option_var_str = ", ".join([str(v) for v in option_vars])
        print(f"    Option Spec: {option.name}({option_var_str})")

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
    traj, all_atoms = trajectory
    assert len(traj.states) == len(all_atoms)
    current_segment_states : List[State] = []
    current_segment_actions : List[Action] = []
    for t in range(len(traj.actions)):
        current_segment_states.append(traj.states[t])
        current_segment_actions.append(traj.actions[t])
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
        if traj.actions[t].has_option():
            # Check for a change in option specs.
            if t < len(traj.actions) - 1:
                option_t = traj.actions[t].get_option()
                option_t1 = traj.actions[t+1].get_option()
                option_t_spec = (option_t.parent, option_t.objects)
                option_t1_spec = (option_t1.parent, option_t1.objects)
                if option_t_spec != option_t1_spec:
                    switch = True
            # Special case: if the final option terminates in the state, we
            # can safely segment without using any continuous info. Note that
            # excluding the final option from the data is highly problematic
            # when using demo+replay with the default 1 option per replay
            # because the replay data which causes no change in the symbolic
            # state would get excluded.
            elif traj.actions[t].get_option().terminal(traj.states[t]):
                switch = True
        if switch:
            # Include the final state as the end of this segment.
            current_segment_states.append(traj.states[t+1])
            current_segment_traj = LowLevelTrajectory(
                current_segment_states, current_segment_actions)
            if traj.actions[t].has_option():
                segment = Segment(current_segment_traj,
                                  all_atoms[t], all_atoms[t+1],
                                  traj.actions[t].get_option())
            else:
                # If option learning, include the default option here; replaced
                # during option learning.
                segment = Segment(current_segment_traj,
                                  all_atoms[t], all_atoms[t+1])
            segments.append(segment)
            current_segment_states = []
            current_segment_actions = []
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
            segment_param_option = DummyOption.parent
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
    # del option_vars

    assert len(params) == len(add_effects) == \
           len(delete_effects) == len(partitions)

    # Prune partitions with not enough data.
    kept_idxs = []
    for idx, partition in enumerate(partitions):
        if len(partition) >= CFG.min_data_for_nsrt:
            kept_idxs.append(idx)
    params = [params[i] for i in kept_idxs]
    add_effects = [add_effects[i] for i in kept_idxs]
    delete_effects = [delete_effects[i] for i in kept_idxs]
    parameterized_options = [parameterized_options[i] for i in kept_idxs]
    option_vars = [option_vars[i] for i in kept_idxs]
    partitions = [partitions[i] for i in kept_idxs]

    # Learn preconditions.
    preconds = [_learn_preconditions(p) for p in partitions]

    # Finalize the operators (with initially empty side effects).
    ops = []
    for i in range(len(params)):
        name = f"Op{i}"
        op = STRIPSOperator(name, params[i], preconds[i], add_effects[i],
                            delete_effects[i], set())
        if verbose:
            print("Learned STRIPSOperator:")
            print(op)
        ops.append(op)

    return ops, partitions, parameterized_options, option_vars


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


def prune_operator_effects(strips_ops: Sequence[STRIPSOperator],
        segmented_trajectories: Sequence[Sequence[Segment]],
        goals: Sequence[Set[GroundAtom]],
        parameterized_options: Sequence[ParameterizedOption],
        option_vars: Sequence[Tuple[Variable]]) -> None:
    """Modifies strips operator side predicates and effects in place.
    """
    # TODO: This is currently a ridiculously slow implementation as a
    # proof-of-concept. There should be many optimizations possible that will
    # hopefully make this not too slow.

    # Find one skeleton per segmented trajectory.
    skeletons = []
    init_atoms = []
    for traj in segmented_trajectories:

        # TODO: make this less stupid.
        objects = set(traj[0].trajectory.states[0])
        ground_ops = []
        for op in strips_ops:
            for ground_op in utils.all_ground_operators(op, objects):
                ground_ops.append(ground_op)

        skeleton = []
        init_atoms.append(traj[0].init_atoms)
        for segment in traj:
            option = segment.get_option()
            for op in utils.get_applicable_operators(ground_ops,
                                                     segment.init_atoms):

                op_idx = strips_ops.index(op.operator)
                param_option = parameterized_options[op_idx]
                opt_vars = option_vars[op_idx]
                if option.parent != param_option:
                    continue
                sub = dict(zip(op.operator.parameters, op.objects))
                ground_opt_vars = [sub[v] for v in opt_vars]
                if list(option.objects) != ground_opt_vars:
                    continue

                pred_atoms = utils.apply_operator(op, segment.init_atoms)
                if pred_atoms == segment.final_atoms:
                    skeleton.append(op)
                    break
            else:
                assert CFG.min_data_for_nsrt > 0
                continue

        skeletons.append(skeleton)

    assert _skeletons_valid(skeletons, init_atoms, goals)

    # Consider pruning each effect from each operator, one at a time.
    # TODO: does the order matter? Hopefully not...
    for strips_op in strips_ops:
        print("Pruning operator:")
        print(strips_op)
        for add_or_delete, effects in (("add", strips_op.add_effects),
                                        ("delete", strips_op.delete_effects)):
            for effect in set(effects):
                # Tentatively remove effect from operator, and check if
                # operators are still valid.
                print(f"Considering removing {effect} from {add_or_delete}")
                effects.remove(effect)
                new_side_predicate = False
                if effect.predicate not in strips_op.side_predicates:
                    strips_op.side_predicates.add(effect.predicate)
                    new_side_predicate = True
                if not _skeletons_valid(skeletons, init_atoms, goals):
                    # Operators are no longer valid, so add back the effect.
                    print("Pruning failed, adding back.")
                    effects.add(effect)
                    if new_side_predicate:
                        strips_op.side_predicates.remove(effect.predicate)
                else:
                    print("Pruning succeeded.")

    assert _skeletons_valid(skeletons, init_atoms, goals)


def _skeletons_valid(skeletons: Sequence[Sequence[STRIPSOperator]],
                     init_atoms: Sequence[Set[GroundAtom]],
                     goals: Sequence[Set[GroundAtom]]) -> bool:
    """Helper for prune_operator_effects.
    """
    assert len(skeletons) == len(init_atoms)

    for skeleton, init_atom_set, goal in zip(skeletons, init_atoms, goals):
        current_atoms = set(init_atom_set)
        for stale_op in skeleton:
            op = stale_op.operator.ground(tuple(stale_op.objects))
            if not op.preconditions.issubset(current_atoms):
                return False
            current_atoms = utils.apply_operator(op, current_atoms)
        if not goal.issubset(current_atoms):
            return False
    return True
