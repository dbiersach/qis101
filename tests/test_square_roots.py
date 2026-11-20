"""Regression checks for the Session 04 square-root examples."""

import importlib.util
import math
import random
import sys
import unittest
from pathlib import Path
from types import ModuleType

from mpmath import mp

SESSION = Path(__file__).resolve().parents[1] / "Session 04 - Computational Mathematics"


def load_example(filename: str) -> ModuleType:
    """Load a course script without running its demonstration."""
    spec = importlib.util.spec_from_file_location(filename[:-3], SESSION / filename)
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BISECTION = load_example("newton_sqrt.py")
HERON = load_example("herons_method.py")
BIG = load_example("big_sqrt.py")


class SquareRootTests(unittest.TestCase):
    """Check relative accuracy, floating-point limits and invalid inputs."""

    def test_float_accuracy_across_scales(self) -> None:
        """Compare all float implementations with the standard-library root."""
        values = [
            0.0,
            math.ulp(0.0),
            sys.float_info.min,
            1e-300,
            1e-250,
            1e-20,
            0.25,
            2.0,
            7.0,
            168923.74,
            1e20,
            1e250,
            1e300,
            sys.float_info.max,
        ]
        rng = random.Random(101)
        values.extend(
            math.ldexp(rng.uniform(1.0, 2.0), rng.randint(-1074, 1023))
            for _ in range(50)
        )
        for function in (BISECTION.newton_sqrt, HERON.newton_sqrt, HERON.heron_sqrt):
            for value in values:
                with self.subTest(
                    function=function.__module__ + "." + function.__name__, value=value
                ):
                    actual = function(value)
                    expected = math.sqrt(value)
                    self.assertTrue(math.isfinite(actual))
                    self.assertTrue(
                        math.isclose(actual, expected, rel_tol=2e-15, abs_tol=0.0)
                    )

    def test_tiny_root_display(self) -> None:
        """Significant-digit output retains a tiny nonzero root."""
        for function in (BISECTION.newton_sqrt, HERON.newton_sqrt, HERON.heron_sqrt):
            displayed = f"{function(1e-250):.16g}"
            self.assertTrue(
                math.isclose(float(displayed), 1e-125, rel_tol=2e-15, abs_tol=0.0)
            )

    def test_float_invalid_inputs(self) -> None:
        """Reject negative and nonfinite real inputs."""
        for function in (BISECTION.newton_sqrt, HERON.newton_sqrt, HERON.heron_sqrt):
            for value in (-1.0, float("inf"), float("-inf"), float("nan")):
                with (
                    self.subTest(function=function.__name__, value=value),
                    self.assertRaises(ValueError),
                ):
                    function(value)

    def test_mpmath_accuracy_across_scales(self) -> None:
        """Check arbitrary-precision results against a higher-precision oracle."""
        values = [
            "1e-500",
            "1e-250",
            "1e-20",
            "0.25",
            "2",
            "168923.74",
            "1e500",
            "3359035138126182262221816387352855681369894766568761568876758902106044097938012929232223664368425159",
        ]
        for precision in (30, 80, 200):
            with mp.workdps(precision):
                self.assertEqual(BIG.sqrt(mp.mpf(0)), 0)
                for text in values:
                    with self.subTest(precision=precision, value=text):
                        value = mp.mpf(text)
                        actual = BIG.sqrt(value)
                        with mp.workdps(precision * 2):
                            expected = mp.sqrt(value)
                            relative_error = abs(actual - expected) / expected
                            self.assertLessEqual(
                                relative_error, mp.mpf(10) ** (2 - precision)
                            )

    def test_mpmath_invalid_inputs(self) -> None:
        """Reject negative and nonfinite arbitrary-precision inputs."""
        for value in (mp.mpf(-1), mp.inf, -mp.inf, mp.nan):
            with self.assertRaises(ValueError):
                BIG.sqrt(value)


if __name__ == "__main__":
    unittest.main()
