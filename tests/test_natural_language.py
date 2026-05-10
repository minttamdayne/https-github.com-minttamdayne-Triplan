from __future__ import annotations

from datetime import date

from src.natural_language import parse_travel_prompt


def test_di_choi_does_not_parse_as_shopping() -> None:
    parsed = parse_travel_prompt(
        "Tôi muốn tới Sài Gòn trong ngày 5/5 tới 8/5, hãy sắp xếp lịch đi chơi",
        fallback_start_date=date(2026, 5, 5),
    )

    assert parsed.start_date == date(2026, 5, 5)
    assert parsed.end_date == date(2026, 5, 8)
    assert "shopping" not in parsed.themes
