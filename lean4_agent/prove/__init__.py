from lean4_agent.util import CONFIG
from lean4_agent.llm import *
from lean4_agent.gym import ProofSearch

from dataclasses import dataclass
from typing import List, Dict

from pydantic import BaseModel

import tiktoken

@dataclass
class ProverState:
    sketch: str
    code: str
    goals: List[str]
    errors: List[Dict]


def code_of_chat_state(state: ChatState, lang="lean"):
    """
    Extracts contents of last lean code block from raw text

    Requires that ChatState.messages[-1] has role "assistant"
    """
    assert state.messages[-1].role == "assistant"

    text = state.messages[-1].content
    left_key = f"```{lang}"
    left_idx = text.rindex(left_key)
    right_idx = text.rindex("```")
    code = text[left_idx + len(left_key) + 1 : right_idx]  # +1 for the newline
    return code.strip()


def sorries_goals_errors_of_lean_state(lean_state):
    sorries = lean_state["sorries"]
    goals = [m["data"] for m in lean_state["messages"] if "unsolved goals" in m["data"]]
    errors = [m for m in lean_state["messages"] if "unsolved goals" not in m["data"]]
    return sorries, goals, errors


def sketch_prompt(code: str) -> str:
    # import that there is a blank line after `by`
    return f"""I am trying to write a proof of the following theorem in Lean 4: 
```lean
{code}
```
I am going to ask you to complete this formal proof. But first, plan out your formal proof in natural language and LaTeX. 

Formatting instructions: enclose your plan in a ```latex code block```"""


def sketch_of_code(code: str, verbose=False) -> str:
    """
    Given a stub of a theorem, returns a natural langauge proof sketch. 

    Args: 
        code (str)
        verbose (bool): whether to print stuff. Defaults to False. 
    """
    user_prompt = sketch_prompt(code)

    chat_state = ChatState(
        messages=[
            ChatMessage(role="system", content=SYSTEM_MESSAGE),
            ChatMessage(role="user", content=user_prompt),
        ]
    )

    chat_state = complete_chat(chat_state)

    if verbose: 
        print(chat_state)

    sketch = code_of_chat_state(chat_state, lang="latex")
    return sketch


def next_tactic_prompt(proverstate: ProverState) -> str:
    goals_string = "\n\n".join(proverstate.goals)
    return f"""I am trying to complete this proof in Lean 4: 
```lean
{proverstate.code}
```

I plan to proceed according to this natural language proof sketch: 

{proverstate.sketch}

The following are the open goals of my code: 
```
{goals_string}
```
1. Please write out a plan for completing the formal proof. Write your plan in English (with LaTeX).
2. Please add the next tactic step to the proof. 

Formatting instructions: include the new version of your (possibly incomplete) proof in a lean code block. Make sure the code block is self-contained and runs. Do not add more than one new tactic step. Do not write `sorry` in your code. Instead, stop writing where you think it will be helpful for me to see the goal state."""


def fix_error_prompt(proverstate: ProverState) -> str:
    error_strings = [
            f'line {x["pos"]["line"]} col {x["pos"]["column"]}:\n{x["data"]}'
            for x in proverstate.errors
            ]

    errors_string = "\n\n".join(error_strings)
    # the newline after the `by` is important
    return f"""The following is a Lean 4 proof I am working on: 

```lean
{proverstate.code}
```

I am following this proof sketch: 
```latex
{proverstate.sketch}
```

This proof returns the following errors. 
```
{errors_string}
```

Please describe how you are going to fix the error. Modify the code to fix the error, but do not add any additional tactic steps.

Formatting instructions: Write the answer in a ```lean code block```. Do not include any sorries in your modified code. Rather, finish writing where you want me to see the goal state."""


def prover_kernel(proverstate: ProverState, mode: str, verbose=False) -> ProverState:
    """
    Takes a ProverState with proverstate.goals nonempty, and prompts a language model
    to fix the error.

    Args:
        proverstate (ProverState)
        mode (str): equal to "prove" or "error"

    Requires:
        if `mode="next_tactic"`, requires `not proverstate.sorries`
        if `model="error"`, requires `proverstate.sorries`
    """
    if mode == "next_tactic":
        assert not proverstate.errors
        user_prompt = next_tactic_prompt(proverstate)
    elif mode == "error":
        assert proverstate.errors
        user_prompt = fix_error_prompt(proverstate)
    else:
        raise ValueError("`mode` not recognized")

    chat_state = ChatState(
        messages=[
            ChatMessage(role="system", content=SYSTEM_MESSAGE),
            ChatMessage(role="user", content=user_prompt),
        ]
    )

    chat_state = complete_chat(chat_state)

    if verbose:
        print(f">>>{mode} MODE")
        print(">>>USER")
        print(chat_state.messages[-2].content)
        print(">>>ASSISTANT")
        print(chat_state.messages[-1].content)

    new_code = code_of_chat_state(chat_state).strip()

    replpath = os.environ.get("PATH_TO_LEAN_REPL")
    lean = ProofSearch(replpath)

    lean_state = lean.run_code(new_code.strip() + "\n", verbose=verbose)

    sorries, goals, errors = sorries_goals_errors_of_lean_state(lean_state)

    # if sorries:
        # raise ValueError("haven't implemented coping with sorries yet")

    new_proverstate = ProverState(
        sketch=proverstate.sketch,
        code=new_code,
        goals=goals,
        errors=errors,
    )

    if verbose:
        print(new_proverstate)

    return new_proverstate, chat_state


def prover(proverstate: ProverState, max_api_calls=10, verbose=False) -> Dict:
    proverstates = [proverstate]
    chat_states = []

    num_api_calls = 0 

    stop_reason = "max_calls"
    for _ in range(max_api_calls):
        if proverstate.errors:
            proverstate, chat_state = prover_kernel(
                proverstate, mode="error", verbose=verbose
            )
        else:
            proverstate, chat_state = prover_kernel(
                proverstate, mode="next_tactic", verbose=verbose
            )

        num_api_calls += 1

        proverstates.append(proverstate)
        chat_states.append(chat_state)

        if not proverstate.errors and not proverstate.goals:
            stop_reason = "done"
            break

    return {
        "proverstates": proverstates,
        "chat_states": chat_states,
        "stop_reason": stop_reason,
        "num_api_calls": num_api_calls
    }

def autoformalize_sketch(code: str, sketch: str, max_api_calls=10, verbose=False) -> Dict:
    replpath = os.environ.get("PATH_TO_LEAN_REPL")
    lean = ProofSearch(replpath)
    lean_state = lean.run_code(code.strip() + "\n", verbose=verbose)
    sorries, goals, errors = sorries_goals_errors_of_lean_state(lean_state)

    proverstate = ProverState(
            sketch=sketch,
            code=code, 
            goals=goals, 
            errors=errors, 
    )

    return prover(proverstate, max_api_calls=max_api_calls, verbose=verbose)

def f2f_prove(code: str, max_api_calls=10, verbose=False) -> Dict:
    sketch = sketch_of_code(code, verbose=verbose)
    return autoformalize_sketch(code, sketch, max_api_calls=max_api_calls, verbose=verbose)
