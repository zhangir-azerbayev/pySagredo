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

import code 

import argparse

def _main():
    """
    Only for debugging/testing purposes
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--autoformalize', action='store_true')
    parser.add_argument('-p', '--prove', action='store_true')
    parser.add_argument('-i', '--input', type=str, help="where to look for input file")
    args = parser.parse_args()

    path = args.input

    start_time = time.time()

    if args.prove: 
        source = open(path).read() 
        summary = f2f_prove(source, max_api_calls=10, verbose=True) 
    
    if args.autoformalize: 
        source = json.load(open(path))
        sketch = source["sketch"]
        code = source["code"]
        summary = autoformalize_sketch(code, sketch, max_api_calls=10, verbose=True)

    end_time = time.time()

    print("\nSUMMARY\nSTOP REASON: ", summary["stop_reason"])
    print(f'{summary["num_api_calls"]} interactions in {end_time-start_time:.2f} seconds')

    #code.interact(local=locals())

    program = summary["proverstates"][-1].code

    print(program)

if __name__ == "__main__":
    _main()
