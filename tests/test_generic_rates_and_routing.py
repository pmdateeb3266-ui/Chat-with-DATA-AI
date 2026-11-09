"""Synthetic tests for generic rate analysis and LLM-free routing.

These cover behaviour that used to be hard-coded to an HR "Attrition"
column and must now work for any two-valued outcome column.
"""
import sys
import types
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "ollama" not in sys.modules:
    sys.modules["ollama"] = types.SimpleNamespace(
        chat=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("An LLM call is not expected in these tests")
        )
    )

from ai_engine import detect_plain_count_question, normalize_result
from data_analyzer import analyze_data, resolve_target_value
from question_router import route_question


def churn_frame():
    return pd.DataFrame({
        "Customer ID": [f"C{i}" for i in range(8)],
        "Plan": ["Basic", "Basic", "Basic", "Basic", "Pro", "Pro", "Pro", "Pro"],
        "Monthly Fee": [10, 10, 12, 12, 30, 30, 35, 35],
        "Churned": ["Yes", "Yes", "Yes", "No", "No", "No", "Yes", "No"],
    })


class RateAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.df = churn_frame()

    def test_overall_rate_infers_positive_value(self):
        result = analyze_data(self.df, {"intent": "rate", "column": "Churned"})
        self.assertTrue(result["success"], result)
        self.assertEqual(result["result_type"], "percentage")
        self.assertEqual(result["value"], 50.0)

    def test_overall_rate_with_explicit_value(self):
        result = analyze_data(
            self.df,
            {"intent": "rate", "column": "churned", "filter_value": "no"},
        )
        self.assertTrue(result["success"], result)
        self.assertEqual(result["value"], 50.0)

    def test_group_rate_ranks_groups(self):
        result = analyze_data(
            self.df,
            {"intent": "group_rate", "column": "Churned", "group_by": "Plan"},
        )
        self.assertTrue(result["success"], result)
        values = result["value"].set_index("Category")["Value"].to_dict()
        self.assertEqual(values, {"Basic": 75.0, "Pro": 25.0})
        self.assertEqual(result["value"].iloc[0]["Category"], "Basic")

    def test_group_rate_without_column_fails_cleanly(self):
        result = analyze_data(self.df, {"intent": "group_rate", "group_by": "Plan"})
        self.assertFalse(result["success"])
        self.assertEqual(result["result_type"], "error")

    def test_rate_on_multi_valued_column_needs_explicit_value(self):
        result = analyze_data(self.df, {"intent": "rate", "column": "Monthly Fee"})
        self.assertFalse(result["success"])

    def test_resolve_target_value_handles_booleans_and_numbers(self):
        self.assertEqual(resolve_target_value(pd.Series([True, False, True])), True)
        self.assertEqual(resolve_target_value(pd.Series([0, 1, 1])), 1)
        self.assertIsNone(resolve_target_value(pd.Series(["A", "B"])))

    def test_old_hr_intents_are_gone(self):
        for intent in ("total_employees", "employees_left", "attrition_rate"):
            with self.subTest(intent=intent):
                result = analyze_data(self.df, {"intent": intent})
                self.assertFalse(result.get("success"), result)


class RateNormalizationTests(unittest.TestCase):
    def test_which_group_has_highest_outcome_becomes_group_rate(self):
        df = churn_frame()
        normalized = normalize_result(
            {"intent": "group_sum", "column": "Monthly Fee", "group_by": "Plan"},
            "Which plan has the highest churned rate?",
            df,
        )
        self.assertEqual(normalized["intent"], "group_rate")
        self.assertEqual(normalized["group_by"], "Plan")
        self.assertEqual(normalized["column"], "Churned")

    def test_loose_column_matching_for_compound_names(self):
        df = pd.DataFrame({
            "JobRole": ["A", "A", "B", "B"],
            "Left_Company": ["Yes", "No", "No", "No"],
        })
        normalized = normalize_result(
            {"intent": "unknown"},
            "Which job role has the highest left company rate?",
            df,
        )
        self.assertEqual(normalized["intent"], "group_rate")
        self.assertEqual(normalized["group_by"], "JobRole")
        self.assertEqual(normalized["column"], "Left_Company")


class RouterTests(unittest.TestCase):
    def test_structural_questions_route_directly(self):
        cases = {
            "How many rows are there?": "row_count",
            "How many records are in the dataset?": "row_count",
            "Number of columns": "column_count",
            "How many columns does this dataset have?": "column_count",
            "How many missing values are there?": "missing_values",
        }
        for question, intent in cases.items():
            with self.subTest(question=question):
                self.assertEqual(route_question(question), {"route": "direct", "intent": intent})

    def test_data_dependent_questions_are_interpreted(self):
        for question in (
            "How many customers churned?",
            "How many rows have missing revenue by region?",
            "Show order count by region",
            "Average unit price by category",
            "What is the churn rate?",
        ):
            with self.subTest(question=question):
                self.assertEqual(route_question(question)["route"], "ollama")

    def test_total_of_a_measure_is_not_a_row_count(self):
        self.assertFalse(detect_plain_count_question("Total revenue"))
        self.assertFalse(detect_plain_count_question("Total sales"))
        self.assertTrue(detect_plain_count_question("Total rows"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
