import fire
import os
import sys
import time
import json
from pathlib import Path
from tqdm import tqdm
from itertools import islice

from pysagredo.util import CONFIG
from pysagredo.llm import *
from pysagredo.gym import ProofSearch
from pysagredo.prove import f2f_prove, autoformalize_sketch

from pydantic import BaseModel

import tiktoken
from datasets import load_dataset

import code 

import argparse

def test_proofsearch(source): 
    print("testing repl...")
    path = os.environ["PATH_TO_LEAN_REPL"]
    print(f"path to repl: {path}")
    proofsearch = ProofSearch(path_to_repl=path)
    out = proofsearch.run_code(source, verbose=True)
    print(out)
    print("repl worked...")

def _main(args):
    """
    Only for debugging/testing purposes
    """

    path = args.input

    start_time = time.time()

    if args.prove and path!="minif2f": 
        source = open(path).read() 
        test_proofsearch(source)
        summary = f2f_prove(source, max_api_calls=10, verbose=True)  

    elif args.prove and path=="minif2f": 
        Path(args.logdir).mkdir(parents=True)
        dataset = load_dataset("hoskinson-center/minif2f-lean4", split="validation")
        for x in tqdm(islice(dataset, 2)): 
            print(x)
            eyed = x["id"]
            source = x["header"] + "\n\n" + x["formal_statement"]
            test_proofsearch(source) # sanity check to make sure repl works
            summary = f2f_prove(source, max_api_calls=10, verbose=True)  
            
            print(f"saving summary for {id}...")
            with open(args.logdir, "w") as f: 
                json.dump(summary, f)
                
    elif args.autoformalize: 
        source = json.load(open(path))
        test_proofsearch(source)
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
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--autoformalize', action='store_true')
    parser.add_argument('-p', '--prove', action='store_true')
    parser.add_argument('-i', '--input', type=str, help="where to look for input file")
    parser.add_argument('-l', '--logdir', type=str, help="directory you want to save logs")
    args = parser.parse_args()
    _main(args)
