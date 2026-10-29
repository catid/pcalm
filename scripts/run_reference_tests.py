#!/usr/bin/env python
"""Dependency-light test runner used before pytest is installed everywhere."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_reference import test_parameter_interval_and_tying, test_causal_shape_and_mask, test_next_token_loss_contract, test_pcalm_state_and_diagnostics, test_pcalm_relax_has_no_global_backward, test_tied_embedding_receives_output_gradient, test_tiny_transformer_parameter_finite_difference
from tests.test_math import test_linear_alm_equilibrium, test_local_linear_finite_difference, test_rank_chunks_are_disjoint_and_counted, test_validation_chunk_isolation
for test in (test_parameter_interval_and_tying, test_causal_shape_and_mask, test_next_token_loss_contract, test_pcalm_state_and_diagnostics, test_pcalm_relax_has_no_global_backward, test_tied_embedding_receives_output_gradient, test_tiny_transformer_parameter_finite_difference, test_linear_alm_equilibrium, test_local_linear_finite_difference, test_rank_chunks_are_disjoint_and_counted, test_validation_chunk_isolation):
    test(); print("PASS", test.__name__)
