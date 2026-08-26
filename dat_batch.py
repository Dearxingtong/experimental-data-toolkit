from __future__ import annotations

from datetime import date


def normalize_case_number(value: str) -> tuple[str | None, str | None]:
    stripped = value.strip()
    if not stripped:
        return None, "Please enter a Case No."
    if not stripped.isdigit():
        return None, "Case No must contain numbers only."

    number = int(stripped)
    if number <= 0:
        return None, "Case No must be greater than zero."

    return f"{number:02d}", None


def format_case_date(value: date) -> str:
    return value.strftime("%y.%m.%d")


def all_workbook_filename(case_number: str, case_date: date) -> str:
    return f"C{case_number}_All_{format_case_date(case_date)}.xlsx"


def position_workbook_filename(
    case_number: str,
    position_number: int,
    case_date: date,
) -> str:
    return f"C{case_number}_P{position_number:02d}_{format_case_date(case_date)}.xlsx"
