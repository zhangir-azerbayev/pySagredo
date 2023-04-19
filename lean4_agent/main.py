import fire
import os
import time
import json

from lean4_agent.util import CONFIG
from lean4_agent.llm import *
from lean4_agent.gym import ProofSearch
from lean4_agent.prove import f2f_prove, autoformalize_sketch

from pydantic import BaseModel

import tiktoken

def _main():
    """
    Only for debugging/testing purposes
    """
    test_case = "evals/f2f/List_append_length.lean"
    # test_case = "evals/autoformalize_proof/imo_1964_p1_2.json"

    source = open(test_case).read() 
    # source = json.load(open(test_case))

    start_time = time.time()
    
    summary = f2f_prove(source, max_api_calls=10, verbose=True) 
    # summary = autoformalize_sketch(**source, max_calls=10)

    end_time = time.time()

    print("STOP REASON: ", summary["stop_reason"])
    print(f'{summary["num_api_calls"]} interactions in {end_time-start_time:.2f} seconds')

    code = summary["proverstates"][-1].code

    print(code)

if __name__ == "__main__":
    _main()
