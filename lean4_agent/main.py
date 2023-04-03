import fire
import os
import time

from lean4_agent.util import CONFIG
from lean4_agent.llm import *
from lean4_agent.gym import ProofSearch
from lean4_agent.prove import f2f_prove

from pydantic import BaseModel

import tiktoken

def _main():
    test_case = "evals/List_append_length.lean"
    source = open(test_case).read()
    
    start_time = time.time()
    summary = f2f_prove(source)
    end_time = time.time()
    num_calls = len(summary["chat"].messages)//2
    print(summary["chat"])
    print("STOP REASON: ", summary["stop_reason"])
    print(f"{num_calls} interactions in {end_time-start_time:.2f} seconds")

if __name__ == "__main__":
    fire.Fire(_main)
