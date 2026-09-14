import pytest

from agent.planner import plan_question, rewrite_query


def test_planner_classifies_regular_news_question() -> None:
    plan = plan_question("What are the risks of rapidly advancing AI?")

    assert plan["intent"] == "news_search"
    assert plan["search_queries"] == ["risks of rapidly advancing AI"]
    assert plan["time_range"] is None


@pytest.mark.parametrize("keyword", ["compare", "versus", "vs", "difference"])
def test_planner_classifies_comparison(keyword: str) -> None:
    plan = plan_question(f"{keyword} solar and wind energy news")

    assert plan["intent"] == "compare_news"


def test_planner_classifies_time_sensitive_question_and_rewrites_query() -> None:
    plan = plan_question("What are the latest AI safety developments?")

    assert plan["intent"] == "time_sensitive_news"
    assert plan["time_range"] == "this_week"
    assert plan["search_queries"] == [
        "latest AI safety developments",
        "AI safety developments recent news",
    ]


def test_rewrites_are_different_for_each_retry() -> None:
    first = rewrite_query("machines becoming dangerously capable", 1)
    second = rewrite_query("machines becoming dangerously capable", 2)

    assert "advanced AI" in first
    assert "safety risks artificial intelligence capabilities" in first
    assert first != second
