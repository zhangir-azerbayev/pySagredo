import fire
import os
from lean4_agent.util import CONFIG
from lean4_agent.llm import generate_message_lean_single

from pydantic import BaseModel

def _main():
    print(generate_message_lean_single("What is a proof of the infinitude of primes?"))

if __name__ == "__main__":
    fire.Fire(_main)
