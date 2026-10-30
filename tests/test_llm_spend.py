"""llm-spend: price logged llm usage from logs.db.

Fixtures build synthetic logs databases with llm's own migrations, in
both storage eras: current `turns` rows and legacy `responses` rows.
With input_tokens=1,000,000 and output_tokens=0, a call costs exactly
its model's per-1M input price.
"""

import json
import sqlite3
from pathlib import Path

import pytest
import sqlite_utils
from click.testing import CliRunner
from llm.cli import cli
from llm.migrations import migrate

import llm_spend
from llm_spend import (
    Price,
    UsageRow,
    aggregate,
    cached_tokens_from_details,
    call_cost,
    load_prices,
)


def _turn(tid, model, inp, out, details=None, dt="2026-10-29T14:00:00+00:00"):
    return {
        "id": tid,
        "thread_id": "th",
        "model": model,
        "resolved_model": model,
        "input_tokens": inp,
        "output_tokens": out,
        "token_details": details,
        "duration_ms": 1,
        "datetime_utc": dt,
    }


def _response(rid, model, inp, out, details=None, dt="2026-10-20T12:00:00+00:00"):
    return {
        "id": rid,
        "model": model,
        "prompt": "p",
        "response": "r",
        "input_tokens": inp,
        "output_tokens": out,
        "token_details": details,
        "duration_ms": 1,
        "datetime_utc": dt,
    }


@pytest.fixture
def user_dir(tmp_path):
    """A fake llm user dir with a seeded logs.db (both eras)."""
    db = sqlite_utils.Database(str(tmp_path / "logs.db"))
    migrate(db)
    db["turns"].insert(_turn("t1", "gpt-5", 2_000_000, 500_000))
    db["turns"].insert(
        _turn(
            "t2",
            "gpt-5",
            1_000_000,
            0,
            details='{"input_tokens_details": {"cached_tokens": 1000000}}',
            dt="2026-10-29T15:00:00+00:00",
        )
    )
    db["turns"].insert(
        _turn("t3", "local-llama", 5000, 800, dt="2026-10-28T09:00:00+00:00")
    )
    db["turns"].insert(
        _turn("t4", "gpt-5-mini", None, None, dt="2026-10-28T10:00:00+00:00")
    )
    db["responses"].insert(
        _response(
            "r1",
            "claude-sonnet-4-5",
            1_000_000,
            200_000,
            details='{"cache_read_input_tokens": 500000}',
        )
    )
    return tmp_path


def _invoke(user_dir, args):
    runner = CliRunner()
    return runner.invoke(
        cli, ["spend", *args], env={"LLM_USER_PATH": str(user_dir)}
    )


def test_cached_token_shapes():
    assert cached_tokens_from_details('{"cache_read_input_tokens": 7}') == 7
    assert cached_tokens_from_details('{"cached_tokens": 8}') == 8
    assert (
        cached_tokens_from_details(
            '{"input_tokens_details": {"cached_tokens": 9}}'
        )
        == 9
    )
    assert (
        cached_tokens_from_details(
            '{"prompt_tokens_details": {"cached_tokens": 10}}'
        )
        == 10
    )
    assert cached_tokens_from_details('{"something_else": 1}') == 0
    assert cached_tokens_from_details("not json") == 0
    assert cached_tokens_from_details(None) == 0


def test_call_cost_with_and_without_cache():
    price = Price(input_per_m=10.0, output_per_m=20.0, cached_input_per_m=1.0)
    plain = UsageRow("m", None, 1_000_000, 500_000, None)
    assert call_cost(plain, price) == pytest.approx(10.0 + 10.0)
    cached = UsageRow(
        "m", None, 1_000_000, 0, '{"cache_read_input_tokens": 1000000}'
    )
    assert call_cost(cached, price) == pytest.approx(1.0)


def test_aggregate_merges_eras_and_marks_unpriced(user_dir):
    rows = llm_spend.load_usage(Path(user_dir) / "logs.db")
    report = aggregate(rows, load_prices(None))
    by_key = {g.key: g for g in report.groups}
    assert report.total_calls == 4
    assert report.calls_without_usage == 1
    assert report.unpriced_models == ["local-llama"]
    # gpt-5: 2 calls; t1 = 2M*1.25 + 500k*10 = 7.50; t2 = 1M cached * 0.125
    assert by_key["gpt-5"].cost == pytest.approx(7.50 + 0.125)
    # claude: 500k*3 + 500k*0.30 + 200k*15 = 1.50 + 0.15 + 3.00
    assert by_key["claude-sonnet-4-5"].cost == pytest.approx(4.65)
    assert by_key["local-llama"].cost is None


def test_aggregate_by_day(user_dir):
    rows = llm_spend.load_usage(Path(user_dir) / "logs.db")
    report = aggregate(rows, load_prices(None), by="day")
    assert [g.key for g in report.groups] == [
        "2026-10-20",
        "2026-10-28",
        "2026-10-29",
    ]


def test_cli_spend_text(user_dir):
    result = _invoke(user_dir, [])
    assert result.exit_code == 0, result.output
    assert "| gpt-5 | 2 |" in result.output
    assert "$7.6250" in result.output
    assert "$12.2750" in result.output
    assert "no price is known for: local-llama" in result.output
    assert "recorded no token usage" in result.output


def test_cli_spend_json(user_dir):
    result = _invoke(user_dir, ["--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["total_calls"] == 4
    assert payload["calls_without_usage"] == 1
    totals = {
        g["key"]: g["cost_usd"] for g in payload["groups"]
    }
    assert totals["gpt-5"] == pytest.approx(7.625)
    assert totals["local-llama"] is None


def test_cli_spend_model_filter(user_dir):
    result = _invoke(user_dir, ["--model", "gpt-5", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [g["key"] for g in payload["groups"]] == ["gpt-5"]


def test_cli_missing_database(tmp_path):
    result = _invoke(tmp_path, [])
    assert result.exit_code == 1
    assert "No logs database" in result.output


def test_cli_empty_database(tmp_path):
    db = sqlite_utils.Database(str(tmp_path / "logs.db"))
    migrate(db)
    result = _invoke(tmp_path, [])
    assert result.exit_code == 0, result.output
    assert "No logged calls" in result.output


def test_prices_override(tmp_path):
    override = tmp_path / "prices.json"
    override.write_text(
        json.dumps(
            {
                "models": {
                    "local-llama": {
                        "input_per_m": 0.0,
                        "output_per_m": 0.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    prices = load_prices(str(override))
    assert prices["local-llama"].input_per_m == 0.0
    assert "gpt-5" in prices  # bundled entries remain


def test_read_only_against_wal_database(user_dir):
    # Opening with mode=ro must not create -wal/-shm sidecars or lock.
    path = Path(user_dir) / "logs.db"
    before = set(Path(user_dir).iterdir())
    llm_spend.load_usage(path)
    after = set(Path(user_dir).iterdir())
    assert before == after
    # And the source DB is untouched.
    db = sqlite3.connect(str(path))
    assert db.execute("select count(*) from turns").fetchone()[0] == 4
