import pandas as pd


# =========================================================
# FIND COLUMN
# =========================================================

def find_column(
    df,
    requested_column
):

    if requested_column is None:
        return None

    requested_column = str(
        requested_column
    ).strip().lower()

    for column in df.columns:

        if (
            str(column)
            .strip()
            .lower()
            == requested_column
        ):

            return column

    return None


# =========================================================
# FIND VALUE
# =========================================================

def find_value(
    series,
    requested_value
):

    if requested_value is None:
        return None

    requested_value = str(
        requested_value
    ).strip().lower()

    for value in series.dropna().unique():

        if (
            str(value)
            .strip()
            .lower()
            == requested_value
        ):

            return value

    return None


# =========================================================
# RESOLVE TARGET VALUE FOR RATE CALCULATIONS
# =========================================================

POSITIVE_VALUES = ("yes", "y", "true", "1", "1.0")


def resolve_target_value(series, requested_value=None):
    """Return the value whose share should be measured.

    Uses the requested value when given. Otherwise, for a
    two-valued column, picks a conventional positive value
    (Yes/True/1). Returns None when the choice is ambiguous.
    """

    if requested_value is not None:
        return find_value(series, requested_value)

    values = series.dropna().unique()

    if len(values) != 2:
        return None

    for value in values:
        if str(value).strip().lower() in POSITIVE_VALUES:
            return value

    return None


# =========================================================
# ANALYZE DATA
# =========================================================

def analyze_data(df, request):

    if request is None:

        return {
            "success": False,
            "answer": "No analysis request was provided.",
            "result_type": "error"
        }

    intent = request.get("intent")


    # =========================================================
    # 1. COLUMN COUNT
    # =========================================================

    if intent == "column_count":

        count = len(df.columns)

        return {
            "success": True,
            "answer": (
                f"The dataset contains "
                f"{count:,} columns."
            ),
            "result_type": "number",
            "value": count
        }


    # =========================================================
    # 2. ROW COUNT
    # =========================================================

    if intent == "row_count":

        count = len(df)

        return {
            "success": True,
            "answer": (
                f"The dataset contains "
                f"{count:,} rows."
            ),
            "result_type": "number",
            "value": count
        }


    # =========================================================
    # 3. MISSING VALUES
    # =========================================================

    if intent == "missing_values":

        total_missing = int(
            df.isnull()
            .sum()
            .sum()
        )

        return {
            "success": True,
            "answer": (
                f"The dataset contains "
                f"{total_missing:,} "
                "missing values."
            ),
            "result_type": "number",
            "value": total_missing
        }


    # =========================================================
    # 4. GENERIC COUNT
    # =========================================================

    if intent == "count":

        count = len(df)

        return {
            "success": True,
            "answer": (
                f"The dataset contains "
                f"{count:,} records."
            ),
            "result_type": "number",
            "value": count
        }


    # =========================================================
    # 5. FILTERED COUNT
    # =========================================================

    if intent == "filtered_count":

        requested_column = request.get(
            "filter_column"
        )

        requested_value = request.get(
            "filter_value"
        )

        actual_column = find_column(
            df,
            requested_column
        )

        if actual_column is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_column}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        actual_value = find_value(
            df[actual_column],
            requested_value
        )

        if actual_value is None:

            return {
                "success": False,
                "answer": (
                    f"The value "
                    f"'{requested_value}' "
                    f"was not found in column "
                    f"'{actual_column}'."
                ),
                "result_type": "error"
            }

        filtered_df = df[
            df[actual_column]
            == actual_value
        ]

        count = len(filtered_df)

        return {
            "success": True,
            "answer": (
                f"There are {count:,} records "
                f"where '{actual_column}' "
                f"is '{actual_value}'."
            ),
            "result_type": "number",
            "value": count,
            "filter_column": actual_column,
            "filter_value": actual_value
        }


    # =========================================================
    # AVERAGE OF ALL NUMERIC COLUMNS
    # =========================================================

    if intent == "average_all_numeric":

        numeric_df = df.select_dtypes(include="number")

        if numeric_df.empty:
            return {
                "success": False,
                "answer": "No numeric columns were found in the dataset.",
                "result_type": "error",
            }

        averages = numeric_df.mean().rename("Average").reset_index()
        averages.columns = ["Column", "Average"]
        averages["Average"] = averages["Average"].round(2)

        return {
            "success": True,
            "answer": (
                f"Calculated the average for "
                f"{len(averages)} numeric columns."
            ),
            "result_type": "table",
            "value": averages,
        }

    # =========================================================
    # 6. AVERAGE
    # =========================================================

    if intent == "average":

        requested_column = request.get(
            "column"
        )

        actual_column = find_column(
            df,
            requested_column
        )

        if actual_column is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_column}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        if not pd.api.types.is_numeric_dtype(
            df[actual_column]
        ):

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{actual_column}' "
                    "is not numeric."
                ),
                "result_type": "error"
            }

        average_value = (
            df[actual_column]
            .mean()
        )

        return {
            "success": True,
            "answer": (
                f"The average of "
                f"'{actual_column}' "
                f"is {average_value:,.2f}."
            ),
            "result_type": "number",
            "value": average_value,
            "column": actual_column
        }


    # =========================================================
    # 7. MAXIMUM
    # =========================================================

    if intent == "maximum":

        requested_column = request.get(
            "column"
        )

        actual_column = find_column(
            df,
            requested_column
        )

        if actual_column is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_column}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        if not pd.api.types.is_numeric_dtype(
            df[actual_column]
        ):

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{actual_column}' "
                    "is not numeric."
                ),
                "result_type": "error"
            }

        maximum_value = (
            df[actual_column]
            .max()
        )

        return {
            "success": True,
            "answer": (
                f"The maximum value of "
                f"'{actual_column}' "
                f"is {maximum_value:,.2f}."
            ),
            "result_type": "number",
            "value": maximum_value,
            "column": actual_column
        }


    # =========================================================
    # 8. MINIMUM
    # =========================================================

    if intent == "minimum":

        requested_column = request.get(
            "column"
        )

        actual_column = find_column(
            df,
            requested_column
        )

        if actual_column is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_column}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        if not pd.api.types.is_numeric_dtype(
            df[actual_column]
        ):

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{actual_column}' "
                    "is not numeric."
                ),
                "result_type": "error"
            }

        minimum_value = (
            df[actual_column]
            .min()
        )

        return {
            "success": True,
            "answer": (
                f"The minimum value of "
                f"'{actual_column}' "
                f"is {minimum_value:,.2f}."
            ),
            "result_type": "number",
            "value": minimum_value,
            "column": actual_column
        }


    # =========================================================
    # DISTINCT VALUES
    # =========================================================

    if intent == "unique_values":

        requested_column = request.get("column")
        actual_column = find_column(df, requested_column)

        if actual_column is None:
            return {
                "success": False,
                "answer": (
                    f"The column '{requested_column}' was not found "
                    "in the dataset."
                ),
                "result_type": "error"
            }

        unique_counts = (
            df[actual_column]
            .value_counts(dropna=False)
            .rename_axis("Category")
            .reset_index(name="Value")
        )

        return {
            "success": True,
            "answer": (
                f"Here are the distinct values in '{actual_column}', "
                "with the number of records for each value."
            ),
            "result_type": "dictionary",
            "value": unique_counts,
            "column": actual_column
        }


    # =========================================================
    # 9. GROUP COUNT
    # =========================================================

    if intent == "group_count":

        requested_group = request.get(
            "group_by"
        )

        actual_group = find_column(
            df,
            requested_group
        )

        if actual_group is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_group}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        grouped = (
            df[actual_group]
            .value_counts(
                dropna=False
            )
            .reset_index()
        )

        grouped.columns = [
            "Category",
            "Value"
        ]

        return {
            "success": True,
            "answer": (
                f"Here is the count by "
                f"'{actual_group}'."
            ),
            "result_type": "dictionary",
            "value": grouped
        }


    # =========================================================
    # 10. GROUP AVERAGE
    # =========================================================

    if intent == "group_average":

        requested_column = request.get(
            "column"
        )

        requested_group = request.get(
            "group_by"
        )

        actual_column = find_column(
            df,
            requested_column
        )

        actual_group = find_column(
            df,
            requested_group
        )

        if actual_column is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_column}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        if actual_group is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_group}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        if not pd.api.types.is_numeric_dtype(
            df[actual_column]
        ):

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{actual_column}' "
                    "is not numeric."
                ),
                "result_type": "error"
            }

        grouped = (
            df.groupby(
                actual_group
            )[actual_column]
            .mean()
            .reset_index()
        )

        grouped.columns = [
            "Category",
            "Value"
        ]

        return {
            "success": True,
            "answer": (
                f"Here is the average of "
                f"'{actual_column}' by "
                f"'{actual_group}'."
            ),
            "result_type": "dictionary",
            "value": grouped
        }


    # =========================================================
    # 11. GROUP SUM
    # =========================================================
    #
    # NEW IN STEP 12K.4
    #
    # Example:
    #
    # Show total purchase amount by category
    #
    # intent:
    # group_sum
    #
    # column:
    # Purchase Amount (USD)
    #
    # group_by:
    # Category
    #
    # Pandas performs the actual calculation.
    # =========================================================

    if intent == "group_sum":

        requested_column = request.get(
            "column"
        )

        requested_group = request.get(
            "group_by"
        )

        actual_column = find_column(
            df,
            requested_column
        )

        actual_group = find_column(
            df,
            requested_group
        )

        if actual_column is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_column}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        if actual_group is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_group}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        if not pd.api.types.is_numeric_dtype(
            df[actual_column]
        ):

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{actual_column}' "
                    "is not numeric."
                ),
                "result_type": "error"
            }

        grouped = (
            df.groupby(
                actual_group
            )[actual_column]
            .sum()
            .reset_index()
        )

        grouped.columns = [
            "Category",
            "Value"
        ]

        return {
            "success": True,
            "answer": (
                f"Here is the total of "
                f"'{actual_column}' by "
                f"'{actual_group}'."
            ),
            "result_type": "dictionary",
            "value": grouped
        }


    # =========================================================
    # 12. RATE / GROUP RATE
    # =========================================================
    # Percentage of rows where `column` equals a target value,
    # overall ("rate") or per group ("group_rate"). Works for
    # any binary or categorical outcome column, e.g. churned,
    # defaulted, returned, approved, Attrition.

    if intent in ("rate", "group_rate"):

        target_column = find_column(df, request.get("column"))

        if target_column is None:
            return {
                "success": False,
                "answer": (
                    "Please specify which column the rate "
                    "should be calculated for."
                ),
                "result_type": "error"
            }

        target_value = resolve_target_value(
            df[target_column],
            request.get("filter_value")
        )

        if target_value is None:
            return {
                "success": False,
                "answer": (
                    f"Please specify which value of "
                    f"'{target_column}' to measure "
                    "(for example 'Yes')."
                ),
                "result_type": "error"
            }

        if len(df) == 0:
            return {
                "success": False,
                "answer": "The dataset is empty.",
                "result_type": "error"
            }

        matches = df[target_column] == target_value

        if intent == "rate":
            rate = float(matches.mean() * 100)
            return {
                "success": True,
                "answer": (
                    f"{rate:.2f}% of records have "
                    f"{target_column} = {target_value} "
                    f"({int(matches.sum()):,} of {len(df):,})."
                ),
                "result_type": "percentage",
                "value": round(rate, 2)
            }

        actual_group = find_column(df, request.get("group_by"))

        if actual_group is None:
            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{request.get('group_by')}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        grouped = (
            matches.groupby(df[actual_group])
            .mean()
            .mul(100)
            .round(2)
            .reset_index()
        )

        grouped.columns = ["Category", "Value"]

        grouped = grouped.sort_values(
            "Value",
            ascending=False
        ).reset_index(drop=True)

        highest_group = grouped.iloc[0]

        return {
            "success": True,
            "answer": (
                f"'{highest_group['Category']}' has the highest "
                f"rate of {target_column} = {target_value} "
                f"at {highest_group['Value']:.2f}%."
            ),
            "result_type": "dictionary",
            "value": grouped
        }


    # =========================================================
    # 13. DISTRIBUTION
    # =========================================================

    if intent == "distribution":

        requested_column = request.get(
            "column"
        )

        actual_column = find_column(
            df,
            requested_column
        )

        if actual_column is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_column}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        return {
            "success": True,
            "answer": (
                f"Here is the distribution of "
                f"'{actual_column}'."
            ),
            "result_type": "distribution",
            "column": actual_column,
            "value": df[
                [actual_column]
            ].copy()
        }


    # =========================================================
    # 14. RELATIONSHIP
    # =========================================================

    if intent == "relationship":

        requested_x = request.get(
            "x_column"
        )

        requested_y = request.get(
            "y_column"
        )

        x_column = find_column(
            df,
            requested_x
        )

        y_column = find_column(
            df,
            requested_y
        )

        if x_column is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_x}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        if y_column is None:

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{requested_y}' "
                    "was not found in the dataset."
                ),
                "result_type": "error"
            }

        if not pd.api.types.is_numeric_dtype(
            df[x_column]
        ):

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{x_column}' "
                    "is not numeric."
                ),
                "result_type": "error"
            }

        if not pd.api.types.is_numeric_dtype(
            df[y_column]
        ):

            return {
                "success": False,
                "answer": (
                    f"The column "
                    f"'{y_column}' "
                    "is not numeric."
                ),
                "result_type": "error"
            }

        relationship_df = (
            df[
                [
                    x_column,
                    y_column
                ]
            ]
            .dropna()
        )

        return {
            "success": True,
            "answer": (
                f"Here is the relationship between "
                f"'{x_column}' and '{y_column}'."
            ),
            "result_type": "relationship",
            "x_column": x_column,
            "y_column": y_column,
            "value": relationship_df
        }


    # =========================================================
    # 15. UNKNOWN
    # =========================================================

    if intent == "unknown":

        return {
            "success": False,
            "answer": (
                "I could not understand the "
                "requested analysis."
            ),
            "result_type": "error"
        }


    # =========================================================
    # 16. ERROR
    # =========================================================

    if intent == "error":

        return {
            "success": False,
            "answer": (
                "There was an error while "
                "understanding the question."
            ),
            "result_type": "error"
        }


    # =========================================================
    # UNSUPPORTED INTENT
    # =========================================================

    return {
        "success": False,
        "answer": (
            "This type of analysis is "
            "not supported yet."
        ),
        "result_type": "error"
    }
