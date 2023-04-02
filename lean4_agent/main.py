import fire
import os
from lean4_agent.util import CONFIG
from lean4_agent.llm import *
from lean4_agent.gym import ProofSearch
from lean4_agent.prove import f2f_prove

from pydantic import BaseModel

import tiktoken

def _main():
    test_case = "evals/Topology_homeomorph.lean"
    source = open(test_case).read()

    summary = f2f_prove(source)
    print(summary["chat"])
    print("STOP REASON: ", summary["stop_reason"])

if __name__ == "__main__":
    fire.Fire(_main)
