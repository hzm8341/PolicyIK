"""Run the official trainer (policyik_sim.train, unmodified) with the Piper tasks registered."""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
import policyik_piper.register  # noqa: F401,E402  (gym registration only)

runpy.run_module("policyik_sim.train", run_name="__main__", alter_sys=True)
