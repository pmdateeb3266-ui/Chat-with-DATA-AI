"""Fast, dataset-agnostic routing for questions that need no LLM.

Only questions about the dataset's structure (rows, columns, missing
values, averages across all numeric columns) are answered directly.
Everything that refers to the data's own columns or values goes to the
interpretation layer in ``ai_engine.understand_question``, which resolves
column names and values from the uploaded dataset.
"""

import re


def normalize_question(question):
    question = str(question).strip().lower()
    question = re.sub(r"\s+", " ", question)
    return question


def _direct(intent):
    return {"route": "direct", "intent": intent}


def _interpret(intent="ai_analysis"):
    return {"route": "ollama", "intent": intent}


# Whole-question patterns only: a question with any extra condition
# (for example "how many rows have missing revenue") must not be
# answered as a plain structural count.
_COLUMN_COUNT = re.compile(
    r"^(?:how many columns(?: are there| does (?:the|this) dataset have)?"
    r"|(?:total )?number of columns|total columns|column count)"
    r"(?: in (?:the|this) dataset)?\??$"
)

_ROW_COUNT = re.compile(
    r"^(?:how many (?:rows|records|entries|observations)"
    r"(?: are there| does (?:the|this) dataset have)?"
    r"|(?:total )?number of (?:rows|records|entries|observations)"
    r"|total (?:rows|records)|row count|record count)"
    r"(?: (?:are )?in (?:the|this) dataset)?\??$"
)

_MISSING = re.compile(
    r"\b(?:missing|null|nan)\s+(?:values|data|cells)\b|\bhow many missing\b"
)

_GROUPED_COUNT = re.compile(
    r"\b(?:count|number) of [\w ]{1,40}? by\b"
    r"|\bhow many [\w ]{1,40}? by\b"
    r"|\b(?:count|distribution) by\b"
)

_GROUPED_AVERAGE = re.compile(r"\b(?:average|avg|mean)\b[\w ()$%-]{0,60}? by\b")

_UNIQUE = re.compile(r"\b(?:unique|distinct) values\b|\bhow many unique\b")


def _asks_average_all_numeric(question):
    return bool(
        re.search(r"\b(?:average|avg|mean)\b", question)
        and re.search(r"\b(?:all|every|each)\b", question)
        and re.search(r"\b(?:numeric|numerical|number)\b", question)
        and re.search(r"\b(?:columns?|fields?|variables?)\b", question)
    )


def route_question(question):
    question = normalize_question(question)

    if _asks_average_all_numeric(question):
        return _direct("average_all_numeric")

    # Breakdowns need the dataset's columns, so they are interpreted.
    if _GROUPED_COUNT.search(question):
        return _interpret("group_count")

    if _GROUPED_AVERAGE.search(question):
        return _interpret("group_average")

    if _COLUMN_COUNT.match(question):
        return _direct("column_count")

    if _ROW_COUNT.match(question):
        return _direct("row_count")

    # A question about missing values in one column ("missing values in
    # Revenue") still returns the dataset-wide summary, which lists every
    # column's missing count.
    if _MISSING.search(question):
        return _direct("missing_values")

    if _UNIQUE.search(question):
        return _interpret("unique_values")

    return _interpret()
