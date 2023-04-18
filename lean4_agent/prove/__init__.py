from lean4_agent.util import CONFIG
from lean4_agent.llm import *
from lean4_agent.gym import ProofSearch

from pydantic import BaseModel

import tiktoken


def code_of_chat_state(state: ChatState):
    """
    Extracts contents of last lean code block from raw text

    Requires that ChatState.messages[-1] has role "assistant"
    """
    text = state.messages[-1].content
    left_key = "```lean"
    left_idx = text.rindex(left_key)
    right_idx = text.rindex("```")
    code = text[left_idx + len(left_key) + 1 : right_idx]  # +1 for the newline
    return code


def sorries_goals_errors_of_lean_state(lean_state):
    sorries = lean_state["sorries"]
    goals = [m["data"] for m in lean_state["messages"] if "unsolved goals" in m["data"]]
    errors = [m for m in lean_state["messages"] if "unsolved goals" not in m["data"]]
    return sorries, goals, errors


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
