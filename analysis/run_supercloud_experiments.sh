#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="analysis/submit.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    COMMON_ARGS="--seed $SEED"

    # cover
    python $FILE --experiment_id cover_prederror --env cover --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function prediction_error
    python $FILE --experiment_id cover_branchfac --env cover --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function branching_factor
    python $FILE --experiment_id cover_energy --env cover --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function lmcut_energy_lookaheaddepth0
    python $FILE --experiment_id cover_count --env cover --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function lmcut_count_lookaheaddepth0
    python $FILE --experiment_id cover_expnodes --env cover --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function expected_nodes

    # blocks
    python $FILE --experiment_id blocks_prederror --env blocks --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function prediction_error
    python $FILE --experiment_id blocks_branchfac --env blocks --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function branching_factor
    python $FILE --experiment_id blocks_energy --env blocks --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function lmcut_energy_lookaheaddepth0
    python $FILE --experiment_id blocks_count --env blocks --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function lmcut_count_lookaheaddepth0
    python $FILE --experiment_id blocks_expnodes --env blocks --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function expected_nodes

    # painting
    python $FILE --experiment_id painting_prederror --env painting --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function prediction_error
    python $FILE --experiment_id painting_branchfac --env painting --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function branching_factor
    python $FILE --experiment_id painting_energy --env painting --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function lmcut_energy_lookaheaddepth0
    python $FILE --experiment_id painting_count --env painting --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function lmcut_count_lookaheaddepth0
    python $FILE --experiment_id painting_expnodes --env painting --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function expected_nodes

    # painting with lid always open
    python $FILE --experiment_id painting_always_open_prederror --painting_lid_open_prob 1.0 --env painting --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function prediction_error
    python $FILE --experiment_id painting_always_open_branchfac --painting_lid_open_prob 1.0 --env painting --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function branching_factor
    python $FILE --experiment_id painting_always_open_energy --painting_lid_open_prob 1.0 --env painting --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function lmcut_energy_lookaheaddepth0
    python $FILE --experiment_id painting_always_open_count --painting_lid_open_prob 1.0 --env painting --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function lmcut_count_lookaheaddepth0
    python $FILE --experiment_id painting_always_open_expnodes --painting_lid_open_prob 1.0 --env painting --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function expected_nodes

    # cluttered_table
    python $FILE --experiment_id ctable_prederror --env cluttered_table --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function prediction_error
    python $FILE --experiment_id ctable_branchfac --env cluttered_table --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function branching_factor
    python $FILE --experiment_id ctable_energy --env cluttered_table --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function lmcut_energy_lookaheaddepth0
    python $FILE --experiment_id ctable_count --env cluttered_table --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function lmcut_count_lookaheaddepth0
    python $FILE --experiment_id ctable_expnodes --env cluttered_table --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_score_function expected_nodes
done
