import pexpect
import json
import argparse

class ProofSearch: 
    def __init__(self, path_to_repl):
        self.proc = pexpect.spawn("lake env lean --run REPL/Main.lean", 
                cwd=path_to_repl)
        self.count = 0 
        
    def run_code(self, code):
        self.proc.sendline(code)
        self.proc.expect_exact(code+"\r\n")
        self.proc.sendline()
        self.proc.expect_exact("\r\n")
        try: 
            index = self.proc.expect("env\": \d+\}", timeout=20)
            output = self.proc.before.decode()
            output += f"env\": {self.count}" + "}"
            print(output)
            return json.loads(output)
        except pexpect.exceptions.TIMEOUT:
            print("FAILED DUE TO TIMEOUT")

def main():
    """For testing purposes"""
    parser = argparse.ArgumentParser(description="main() for testing purposes only")

    parser.add_argument("--replpath", type=str, required=True, 
            help="path to leanprover-community/repl")

    args = parser.parse_args()

    proofsearch = ProofSearch(args.replpath)
    proofsearch.run_code("{ \"cmd\" : \"import Mathlib.Data.List.Basic\\ndef f := 2\" }")
    proofsearch.run_code("{ \"cmd\" : \"example : 2 = 3 := by\" }")
    proofsearch.run_code("{ \"cmd\" : \"example : 2 = 3 := rfl\" }")

if __name__=="__main__":
    main()
