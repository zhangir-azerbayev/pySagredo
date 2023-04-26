import unittest
from pysagredo.gym import ProofSearch
import os


class TestGym(unittest.TestCase):
    def test_run_code(self): 
        if "PATH_TO_LEAN_REPL" in os.environ:
            path = os.environ.get("PATH_TO_LEAN_REPL")
        else: 
            raise EnvironmentError("no PATH_TO_LEAN_REPL")

        print(f"path to repl: {path}")
        proofsearch = ProofSearch(path_to_repl=path)

        # should return empty sorries and goals
        out = proofsearch.run_code("def f := 2", verbose=True)
        self.assertEqual(out["sorries"], [])
        self.assertEqual(out["messages"], [])

        # should return goal state
        out = proofsearch.run_code("example : 2 = 2 := by", verbose=True)
        self.assertTrue(any("unsolved goals" in m["data"] for m in out["messages"]))

        # should return error
        out = proofsearch.run_code("example : 2 = 3 := rfl", verbose=True)
        self.assertTrue("type mismatch" in out["messages"][0]["data"])

        # should return goal state
        feedback = proofsearch.run_code("def f := 37", verbose=True)
        env = feedback["env"]
        out = proofsearch.run_code("#check (rfl: f = 37)", env=env, verbose=True)
        print(out)
        self.assertTrue(all(m["severity"]!="error" for m in out["messages"]))

