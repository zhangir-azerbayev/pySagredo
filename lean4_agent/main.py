import fire
import os
from lean4_agent.util import CONFIG
from lean4_agent.llm import *
from lean4_agent.gym import ProofSearch

from pydantic import BaseModel

PATH_TO_REPL = "/home/zhangir/projects/repl"

def code_of_state(state: ChatState): 
    """
    Extracts contents of last lean code block from raw text

    Requires that ChatState.messages[-1] has role "assistant"
    """
    text = state.messages[-1].content 
    left_key = "```lean"
    left_idx = text.rindex(left_key)
    right_idx = text.rindex("```")
    code = text[left_idx+len(left_key)+1:right_idx] # +1 for the newline
    return code

def f2f_prove(source: str, max_calls=6):
    """
    Currently incomplete. Exits after one response. 

    `source` is an incomplete Lean proof, which consists of imports/namespace, 
    and `{theorem statement} := by`
    """
    
    lean = ProofSearch(PATH_TO_REPL)

    initial_prompt = ChatState(
        messages=[
            ChatMessage(role="system", content=SYSTEM_MESSAGE),
            ChatMessage(role="user", content=f2f_initial_prompt(source)),
        ]
    )

    state = complete_chat(initial_prompt)

    code = code_of_state(state).strip()
    print(f"EXTRACTED CODE:\n{repr(code)}")
    feedback = lean.run_code("{ \"cmd\" : \"" + repr(code) + "\" }")
    print(feedback)

def _main():
    test_case = "tests/List_append_length.lean"
    source = open(test_case).read()

    f2f_prove(source)


if __name__ == "__main__":
    fire.Fire(_main)
