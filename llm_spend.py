"""llm-spend: price what your `llm` history actually cost.

An `llm` plugin that adds one command:

    llm spend

`llm` already logs every response's token usage to its SQLite database
(logs.db) -- input tokens, output tokens, and provider token details --
but it never turns those numbers into dollars. This plugin reads that
database (read-only), prices each logged call against a small bundled
price table (overridable with --prices), and reports estimated spend
grouped by model or by day.

Two storage eras are supported: current `llm` versions log usage to the
`turns` table; older versions used `responses`. A call is written to
exactly one of them, so both are read and merged.

Prices are EXAMPLE data (see PRICES_AS_OF): providers change prices
often, and models logged under aliases or local endpoints may have no
price at all -- those calls are reported with their token counts and an
unknown cost, never silently priced at $0.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import click
import llm
from llm import hookimpl

PRICES_AS_OF = "2026-10-30"

# USD per 1M tokens: (input, output, cached_input or None).
BUNDLED_PRICES: dict[str, tuple[float, float, Optional[float]]] = {
    "gpt-5": (1.25, 10.00, 0.125),
    "gpt-5-mini": (0.25, 2.00, 0.025),
    "gpt-4o": (2.50, 10.00, 1.25),
    "gpt-4o-mini": (0.15, 0.60, 0.075),
    "claude-sonnet-4-5": (3.00, 15.00, 0.30),
    "claude-haiku-4-5": (1.00, 5.00, 0.10),
    "gemini-2.5-pro": (1.25, 10.00, 0.31),
    "gemini-2.5-flash": (0.30, 2.50, 0.075),
}


@dataclass(frozen=True)
class Price:
    input_per_m: float
    output_per_m: float
    cached_input_per_m: Optional[float] = None


def load_prices(path: Optional[str]) -> dict[str, Price]:
    prices = {
        model: Price(*rates) for model, rates in BUNDLED_PRICES.items()
    }
    if path is not None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = data.get("models", data)
        for model, entry in entries.items():
            prices[model] = Price(
                input_per_m=float(entry["input_per_m"]),
                output_per_m=float(entry["output_per_m"]),
                cached_input_per_m=(
                    float(entry["cached_input_per_m"])
                    if entry.get("cached_input_per_m") is not None
                    else None
                ),
            )
    return prices


@dataclass
class UsageRow:
    model: str
    datetime_utc: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    token_details: Optional[str]


def _read_table(db: sqlite3.Connection, table: str) -> list[UsageRow]:
    columns = {
        row[1] for row in db.execute(f"PRAGMA table_info({table})")
    }
    if not columns:
        return []
    model_expr = (
        "coalesce(resolved_model, model)"
        if "resolved_model" in columns
        else "model"
    )
    rows = db.execute(
        f"SELECT {model_expr}, datetime_utc, input_tokens, output_tokens,"
        f" token_details FROM {table}"
    ).fetchall()
    return [
        UsageRow(
            model=model,
            datetime_utc=dt,
            input_tokens=inp,
            output_tokens=out,
            token_details=details,
        )
        for model, dt, inp, out, details in rows
    ]


def load_usage(db_path: Path) -> list[UsageRow]:
    """Read usage rows from both storage eras, read-only."""
    uri = f"file:{db_path}?mode=ro"
    db = sqlite3.connect(uri, uri=True)
    try:
        tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        rows: list[UsageRow] = []
        for table in ("turns", "responses"):
            if table in tables:
                rows.extend(_read_table(db, table))
        return rows
    finally:
        db.close()


def cached_tokens_from_details(token_details: Optional[str]) -> int:
    """Best-effort extraction of cached input tokens from the provider
    token-details JSON `llm` stores. Shapes seen in the wild:
    {"cache_read_input_tokens": N}, {"cached_tokens": N},
    {"input_tokens_details": {"cached_tokens": N}},
    {"prompt_tokens_details": {"cached_tokens": N}}.
    Anything unrecognized yields 0 (those tokens are then priced at the
    full input rate, never double-discounted).
    """
    if not token_details:
        return 0
    try:
        details = json.loads(token_details)
    except (TypeError, ValueError):
        return 0
    if not isinstance(details, dict):
        return 0
    for key in ("cache_read_input_tokens", "cached_tokens"):
        value = details.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    for key in ("input_tokens_details", "prompt_tokens_details"):
        nested = details.get(key)
        if isinstance(nested, dict):
            value = nested.get("cached_tokens")
            if isinstance(value, int) and value >= 0:
                return value
    return 0


def call_cost(row: UsageRow, price: Price) -> float:
    """USD cost of one logged call."""
    input_tokens = row.input_tokens or 0
    output_tokens = row.output_tokens or 0
    cached = min(cached_tokens_from_details(row.token_details), input_tokens)
    cached_rate = (
        price.cached_input_per_m
        if price.cached_input_per_m is not None
        else price.input_per_m
    )
    return (
        (input_tokens - cached) * price.input_per_m
        + cached * cached_rate
        + output_tokens * price.output_per_m
    ) / 1_000_000.0


@dataclass
class Group:
    key: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: Optional[float] = 0.0  # None when any call in the group is unpriced

    def add(self, row: UsageRow, price: Optional[Price]) -> None:
        self.calls += 1
        self.input_tokens += row.input_tokens or 0
        self.output_tokens += row.output_tokens or 0
        if price is None:
            self.cost = None
        elif self.cost is not None:
            self.cost += call_cost(row, price)


@dataclass
class SpendReport:
    groups: list[Group] = field(default_factory=list)
    total_calls: int = 0
    calls_without_usage: int = 0
    unpriced_models: list[str] = field(default_factory=list)


def aggregate(
    rows: list[UsageRow],
    prices: dict[str, Price],
    by: str = "model",
    model_filter: Optional[str] = None,
) -> SpendReport:
    report = SpendReport()
    groups: dict[str, Group] = {}
    unpriced: set[str] = set()
    for row in rows:
        if row.input_tokens is None and row.output_tokens is None:
            report.calls_without_usage += 1
            continue
        if model_filter is not None and row.model != model_filter:
            continue
        report.total_calls += 1
        price = prices.get(row.model)
        if price is None:
            unpriced.add(row.model)
        if by == "day":
            key = (row.datetime_utc or "unknown")[:10]
        else:
            key = row.model
        group = groups.setdefault(key, Group(key=key))
        group.add(row, price)
    if by == "day":
        report.groups = [groups[k] for k in sorted(groups)]
    else:
        # Priced groups first (most expensive first), unpriced last.
        report.groups = sorted(
            groups.values(),
            key=lambda g: (
                g.cost is None,
                -(g.cost or 0.0),
                g.key,
            ),
        )
    report.unpriced_models = sorted(unpriced)
    return report


def _money(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"${value:,.4f}"


def render_text(report: SpendReport, by: str) -> str:
    lines = []
    header = ("Date" if by == "day" else "Model")
    lines.append(
        f"| {header} | Calls | Input tokens | Output tokens | Est. cost |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    total_cost = 0.0
    any_unpriced = False
    total_in = 0
    total_out = 0
    for g in report.groups:
        lines.append(
            f"| {g.key} | {g.calls} | {g.input_tokens:,} | "
            f"{g.output_tokens:,} | {_money(g.cost)} |"
        )
        total_in += g.input_tokens
        total_out += g.output_tokens
        if g.cost is None:
            any_unpriced = True
        else:
            total_cost += g.cost
    lines.append(
        f"| **Total** | **{report.total_calls}** | **{total_in:,}** | "
        f"**{total_out:,}** | **{_money(total_cost)}** |"
    )
    notes = []
    if any_unpriced:
        notes.append(
            "Total covers priced models only; no price is known for: "
            + ", ".join(report.unpriced_models)
            + " (add one with --prices)."
        )
    if report.calls_without_usage:
        notes.append(
            f"{report.calls_without_usage} logged call(s) recorded no "
            "token usage and are excluded."
        )
    notes.append(
        f"Prices are example data as of {PRICES_AS_OF}; providers change "
        "prices often -- verify before relying on the totals."
    )
    lines += ["", *[f"_{n}_" for n in notes]]
    return "\n".join(lines) + "\n"


def report_to_dict(report: SpendReport, by: str) -> dict:
    return {
        "by": by,
        "prices_as_of": PRICES_AS_OF,
        "total_calls": report.total_calls,
        "calls_without_usage": report.calls_without_usage,
        "unpriced_models": report.unpriced_models,
        "groups": [
            {
                "key": g.key,
                "calls": g.calls,
                "input_tokens": g.input_tokens,
                "output_tokens": g.output_tokens,
                "cost_usd": (
                    round(g.cost, 6) if g.cost is not None else None
                ),
            }
            for g in report.groups
        ],
    }


@hookimpl
def register_commands(cli):
    @cli.command(name="spend")
    @click.option(
        "--database",
        type=click.Path(exists=True, dir_okay=False),
        default=None,
        help="Path to logs.db (default: llm's own logs database)",
    )
    @click.option(
        "--days",
        type=int,
        default=None,
        help="Only include calls from the last N days",
    )
    @click.option(
        "--by",
        type=click.Choice(["model", "day"]),
        default="model",
        help="Group spend by model or by day (default: model)",
    )
    @click.option(
        "--model", "model_filter", default=None,
        help="Only include calls to this model",
    )
    @click.option(
        "--prices",
        type=click.Path(exists=True, dir_okay=False),
        default=None,
        help="JSON price file overriding/extending the bundled table",
    )
    @click.option(
        "--json", "as_json", is_flag=True,
        help="Emit the report as JSON",
    )
    def spend(database, days, by, model_filter, prices, as_json):
        """Estimate what your logged llm usage cost, from logs.db."""
        db_path = (
            Path(database) if database else llm.user_dir() / "logs.db"
        )
        if not db_path.exists():
            raise click.ClickException(
                f"No logs database at {db_path} -- run a prompt with "
                "`llm` first (see `llm logs path`)."
            )
        rows = load_usage(db_path)
        if days is not None:
            from datetime import datetime, timedelta, timezone

            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=days)
            ).isoformat()
            rows = [
                r
                for r in rows
                if r.datetime_utc is not None and r.datetime_utc >= cutoff
            ]
        report = aggregate(
            rows, load_prices(prices), by=by, model_filter=model_filter
        )
        if as_json:
            click.echo(json.dumps(report_to_dict(report, by), indent=2))
        else:
            if report.total_calls == 0 and report.calls_without_usage == 0:
                click.echo("No logged calls found in this database.")
            else:
                click.echo(render_text(report, by), nl=False)
