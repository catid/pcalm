#!/usr/bin/env python
"""Dependency-light test runner used before pytest is installed everywhere."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_reference import test_parameter_interval_and_tying, test_causal_shape_and_mask, test_next_token_loss_contract, test_pcalm_state_and_diagnostics
for test in (test_parameter_interval_and_tying, test_causal_shape_and_mask, test_next_token_loss_contract, test_pcalm_state_and_diagnostics):
    test(); print("PASS", test.__name__)
