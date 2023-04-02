import requests
from requests import HTTPError
from dataclasses import dataclass
from typing import List
from tiktoken import get_encoding
import os
from pydantic import BaseModel
import backoff
import tiktoken

SYSTEM_MESSAGE: str = """\
You are a pure mathematician who is an expert in the Lean 4 theorem prover. Your job is help your user write Lean proofs.
I want to remind you that we're using Lean 4, not the older Lean 3, and there have been some syntax changes. In particular:
- Type constants are now UpperCamelCase, eg `Nat`, `List`.
- Term constants and variables are now `lowerCamelCase` rather than `snake_case`. For example, we now have `NumberTheory.Divisors.properDivisors instead of `number_theory.divisors.proper_divisors`.
- Pure functions are now written with the syntax `fun x => f x`. The old `λ x, f x` syntax will not work.
- Instead of being separated by a comma, tactics can be separated by a newline or by a semicolon. For example, we could write
```lean
theorem test (p q : Prop) (hp : p) (hq : q) : p ∧ q ∧ p := by
  apply And.intro hp
  exact And.intro hq hp
```
or
```lean
theorem test (p q : Prop) (hp : p) (hq : q) : p ∧ q ∧ p := by
  apply And.intro hp; exact And.intro hq hp
```
- Indentation is significant.
- In the `rw` tactic you must enclose the lemmas in square brackets, even if there is just one. For example `rw h1` is now `rw [h1]`.
- The `induction` tactic now uses a structured format, like pattern matching. For example, in Lean 4 we can write
```lean
theorem zero_add (n : Nat) : 0 + n = n := by
  induction n with
  | zero => rfl
  | succ n ih => rw [Nat.add_succ, ih]
```
  Alternatively you can still use `induction' with x y ih`, like in Lean 3.
- The `cases` tactic now uses a structured format, like pattern matching. For example, in Lean 4 we can write
```lean
example (p q : Prop) : p ∨ q → q ∨ p := by
  intro h
  cases h with
  | inl hp => apply Or.inr; exact hp
  | inr hq => apply Or.inl; exact hq\
"""

PROOF_INSTRUCTION = """\
1. Please write out a plan for proceeding with the proof. Write your plan in English (with LaTeX).
2. Please add the next tactic step to the proof. Include the new version of your (possibly incomplete) proof in a lean code block. Make sure the code block is self-contained and runs. Do not add more than one new tactic step."""

def f2f_initial_prompt(code): 
    return f"""\
I am going to show you an incomplete proof and the accompanying goal state. I will ask you to complete the proof step by step, adding one tactic step in each response. 

Here is my Lean code so far: 
```lean
{code}
```
{PROOF_INSTRUCTION}"""

def autoformalize_proof_initial_prompt(nl_statement, nl_proof, code):
    return f"""\
I am going to show you a natural language proof of a theorem and a corresponding formal theorem statement in Lean 4. Your job will be to write a formal proof of the formal theorem statement, using the natural language proof as a hint.

Here are the natural language theorem and proof:
\\begin{theorem}
    {nl_statement}
\\end{theorem}
{nl_proof}

Below is the Lean code I would like you to complete.
```lean
{code}
```
{PROOF_INSTRUCTION}"""

def autoformalize_statement_and_proof_initial_prompt(nl_statement, nl_proof, code): 
    return f"""\
I am going to show you a natural language theorem statement and natural language proof of that theorem. Your job will be to formalize the statement of the theorem in Lean 4 and formally prove the statement. 

Here are the natural language theorem and proof:
\\begin{theorem}
    {nl_statement}
\\end{theorem}
{nl_proof}

Here is the code template for your formalization. 
```lean
{code}
```
1. Please write out a plan for the autoformalization. Write your plan in English (with LaTeX).
2. Autoformalize the statement of the theorem only, i.e, remove the `sorry` in the type signature of the theorem. Do not start proving the theorem yet. Include the new version of your code in a lean code block. Make sure your code is self-contained and runs.
"""

def prove_unsolved_goals_prompt(goal_state):
    return f"""\
Here is the new goal state:
```lean
{goal_state}
```
{PROOF_INSTRUCTION}"""


def prove_error_prompt(error): 
    return f"""\
Executing this code returns errors:
```
{error}
```
Please describe how you are going to fix the error. Modify your code to fix the error, but do not add any new tactic steps.
"""

def sorry_prompt():
    return """\
There is a sorry in your code. Please do not write any code that contains sorries. Instead, finish typing at the location where you want to see the goal state. Remove the sorry, but do not add any new tactic steps.\
"""

class ChatMessage(BaseModel):
    role: str
    content: str

    def __str__(self): 
        return f">>>{self.role.upper()}\n" + self.content

class ChatState(BaseModel):
    messages: List[ChatMessage]

    def __str__(self): 
        return "\n".join(str(x) for x in self.messages)

@backoff.on_exception(backoff.expo, HTTPError)
def generate_message(chat_state: ChatState, temperature=0, max_tokens=2048, model: str = "gpt-4") -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    }
    payload = dict(
        model=model,
        messages=chat_state.dict()["messages"],
        max_tokens=max_tokens,
        stream=False,
        temperature=temperature,
    )
    r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if r.status_code != 200: 
        print(chat_state)

        enc = tiktoken.encoding_for_model(model)
        num_tokens = len(enc.encode(str(chat_state)))
        print(f"Chat contains approx {num_tokens} tokens")

        print(r, "retrying")

        raise HTTPError

    return r.json()["choices"][0]["message"]["content"]

def complete_chat(chat_state: ChatState, **kwargs): 
    print("waiting on api...")
    response_text = generate_message(chat_state, **kwargs)
    print(f"GOT RESPONSE")
    return ChatState(messages=[*chat_state.messages, ChatMessage(role="assistant", content=response_text)])

def generate_message_lean_single(input: str):
    return generate_message(ChatState(messages=[ChatMessage(role="system", content=SYSTEM_MESSAGE), ChatMessage(role="user", content=input)]))

