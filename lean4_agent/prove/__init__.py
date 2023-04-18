from lean4_agent.util import CONFIG
from lean4_agent.llm import *
from lean4_agent.gym import ProofSearch

from dataclasses import dataclass
from typing import List

from pydantic import BaseModel

import tiktoken

@dataclass 
class Sorry: 
    line: int
    column: int
    text: int

@dataclass Goal: 
    text: int 

@dataclass Error: 
    text: int

@dataclass
class ProverState: 
    sketch: str
    code: str 
    sorries: List[Sorry]
    goals: List[Goal]
    errors: List[Error]

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
    return code


def sorries_goals_errors_of_lean_state(lean_state):
    sorries = lean_state["sorries"]
    goals = [m["data"] for m in lean_state["messages"] if "unsolved goals" in m["data"]]
    errors = [m for m in lean_state["messages"] if "unsolved goals" not in m["data"]]
    return sorries, goals, errors

def sketch_prompt(code: str) -> str: 
    return f"""\
I am trying to write a proof of the following theorem in Lean 4: 
```lean
{code}
```
I am going to ask you to complete this formal proof. But first, plan out your formal proof in natural language and LaTeX. 

Formatting instructions: enclose your plan in a ```code block```.\
"""

def sketch_of_code(code: str) -> str: 
    user_prompt = sketch_prompt(code)

    chat_state = ChatState(
        messages=[
            ChatMessage(role="system", content=SYSTEM_MESSAGE),
            ChatMessage(role="user", content=user_prompt),
        ]
    )

    chat_state = complete_chat(chat_state)

    sketch = code_of_chat_state(chat_state, lang="")
    return sketch

def prove_prompt(proverstate: ProverState) -> str: 
    return f"""\
I am trying to complete this proof in Lean 4: 
```lean
{proverstate.code}
```

I plan to proceed according to this natural language proof sketch: 

{proverstate.sketch}

The following are the open goals of my code: 
```
{"\n\n".join([x.text for x in proverstate.goals])}
```
1. Please write out a plan for completing the formal proof. Write your plan in English (with LaTeX).
2. Please add the next tactic step to the proof. 

Formatting instructions: include the new version of your (possibly incomplete) proof in a lean code block. Make sure the code block is self-contained and runs. Do not add more than one new tactic step. Do not write `sorry` in your code. Instead, stop writing where you think it will be helpful for me to see the goal state. 
"""

def fix_error_prompt(proverstate: ProverState) -> str: 
    return f"""\
The following is a Lean 4 proof I am working on: 

```lean
{proverstate.code}
```

I am following this proof sketch: 

{proverstate.sketch}

This proof returns the following errors. 
```
{"\n\n".join([x.text for x in proverstate.errors])}
```

Please describe how you are going to fix the error. Modify the code to fix the error, but do not add any additional tactic steps. 

Formatting instructions: Write the answer in a `lean` code block. Do not include any sorries in your modified code. Rather, finish writing where you want me to see the goal state.\
"""

def prover_kernel(proverstate: ProverState, mode: str, verbose=False) -> ProverState: 
    """
    Takes a ProverState with proverstate.goals nonempty, and prompts a language model 
    to fix the error.

    Args: 
        proverstate (ProverState)
        mode (str): equal to "prove" or "error"

    Requires: 
        if `mode="prover"`, requires `not proverstate.sorries`
        if `model="error"`, requires `proverstate.sorries`
    """
    if mode="prover":
        assert not proverstate.errors
    elif model="error": 
        assert proverstate.errors
    else: 
        raise ValueError("`mode` not recognized")

    user_prompt = fix_error_prompt(proverstate)

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

    lean_state = lean.run_code(new_code, verbose=verbose)
    lean_states.append(lean_state)

    sorries, goals, errors = sorries_goals_errors_of_lean_state(lean_state)

    if sorries: 
        raise ValueError("haven't implemented coping with sorries yet")

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
    pass

def f2f_prove(code: str, **kwargs) -> Dict: 
    pass
    

"""
below this line is all old stuff
"""

def calc_tokens(chat_state: ChatState, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    num_tokens = len(enc.encode(str(chat_state)))
    return num_tokens


def chat_prove_kernel(lean: ProofSearch, chat_state: ChatState, sorries, goals, errors):
    if goals and not errors:
        goal_state = "\n\n".join(goals)
        prompt = prove_unsolved_goals_prompt(goal_state)
    elif errors:
        all_errors = "\n\n".join(
            [
                f'line {m["pos"]["line"]} column {m["pos"]["column"]}:\n' + m["data"]
                for m in errors
            ]
        )
        prompt = prove_error_prompt(all_errors)
    elif sorries:
        prompt = sorry_prompt()
    else:
        print("warning, `chat_prove_kernel` doing nothing")
        return chat_state, sorries, goals, errors, None

    chat_state = ChatState(
        messages=[*chat_state.messages, ChatMessage(role="user", content=prompt)]
    )
    chat_state = complete_chat(chat_state)

    code = code_of_chat_state(chat_state).strip()

    print("waiting for lean server...")
    lean_state = lean.run_code(code, verbose=False)
    sorries, goals, errors = sorries_goals_errors_of_lean_state(lean_state)

    return chat_state, sorries, goals, errors, lean_state


def prove_with_initial_prompt(initial_prompt: str, max_calls=6, max_tokens=8192):
    """
    `initial_prompt` should ask the model to generate a response contained within
    a lean code block that ends in `:= by`.
    """
    replpath = os.environ.get("PATH_TO_LEAN_REPL")
    lean_states = []
    stop_reason = "max_calls"

    chat_state = ChatState(
        messages=[
            ChatMessage(role="system", content=SYSTEM_MESSAGE),
            ChatMessage(role="user", content=initial_prompt),
        ]
    )

    lean = ProofSearch(replpath)

    chat_state = complete_chat(chat_state)
    code = code_of_chat_state(chat_state).strip()

    lean_state = lean.run_code(code, verbose=True)
    lean_states.append(lean_state)

    sorries, goals, errors = sorries_goals_errors_of_lean_state(lean_state)

    if sorries or goals or errors:
        for num_calls in range(1, max_calls):
            num_tokens = calc_tokens(chat_state)
            if num_tokens > max_tokens:
                stop_reason = "max_tokens"

            del lean
            lean = ProofSearch(replpath)

            chat_state, sorries, goals, errors, lean_state = chat_prove_kernel(
                lean, chat_state, sorries, goals, errors
            )
            lean_states.append(lean_state)
            if not (sorries or goals or errors):
                stop_reason = "success"
                break
    else:
        stop_reason = "success"

    summary = {
        "code": code,
        "chat": chat_state,
        "lean_states": lean_states,
        "stop_reason": stop_reason,
    }
    return summary


def f2f_prove(code: str, **kwargs):
    return prove_with_initial_prompt(f2f_initial_prompt(code), **kwargs)

def autoformalize_proof(nl_statement: str, nl_proof: str, code: str, **kwargs):
    return prove_with_initial_prompt(
        autoformalize_proof_initial_prompt(nl_statement, nl_proof, code), 
        **kwargs
    )

def autoformalize_statement_and_proof(
    nl_statement: str, nl_proof: str, code: str, **kwargs
):
    return prove_with_initial_prompt(
        autoformalize_statement_and_proof_initial_prompt(nl_statement, nl_proof, code),
        **kwargs,
    )


from capabilities import Capability

synth = Capability("multi/structured")

TOY_PROMPT: str = """\
    I am going to show you a natural language proof of a theorem and a corresponding formal theorem statement in Lean 4. Your job will be to write a formal proof of the formal theorem statement, using the natural language proof as a hint.

Here are the natural language theorem and proof:
\\begin\{theorem\}
    Show that for any natural number $n$, $7$ does not divide $2^n + 1$.
\\end\{theorem\}
If $2^n+1$ is congruent to 0 mod 7, then $2^n$ must be congruent to 6 mod 7, but this is not possible due to how $2^n$ mod 7 cycles. Therefore, there is no solution.

Plan out a plan for your formal proof. You can use the natural language proof as a guide, but there is no need to follow it exactly, or at all. Return this as a list of strings called `proof_steps`, where each proof step is a complete sentence and each proof step logically follows from the previous one.
"""

from pydantic import BaseModel

class Metadata(BaseModel):
    theorem: str

class InformalProof(BaseModel):
    proof_steps: List[str]

inp = Metadata(theorem="theorem imo_1964_p1_2 (n : ℕ) : ¬ 7 ∣ (2^n + 1)")

instructions = TOY_PROMPT


if __name__ == "__main__":
    print(synth(Metadata, InformalProof, instructions, inp))
