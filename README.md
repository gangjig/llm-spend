# llm-spend

**Price what your [`llm`](https://llm.datasette.io/) history actually cost.**

`llm` already logs every response's token usage to its SQLite database
(`logs.db`) — input tokens, output tokens, provider token details — but
it never turns those numbers into dollars. This plugin adds one command
that does:

```bash
llm install llm-spend
llm spend
```

```
| Model | Calls | Input tokens | Output tokens | Est. cost |
|---|---:|---:|---:|---:|
| gpt-5 | 2 | 3,000,000 | 500,000 | $7.6250 |
| claude-sonnet-4-5 | 1 | 1,000,000 | 200,000 | $4.6500 |
| local-llama | 1 | 5,000 | 800 | — |
| **Total** | **4** | **4,005,000** | **700,800** | **$12.2750** |

_Total covers priced models only; no price is known for: local-llama (add one with --prices)._
_1 logged call(s) recorded no token usage and are excluded._
```

## Usage

```bash
llm spend                          # all logged calls, grouped by model
llm spend --by day                 # grouped by day instead
llm spend --days 30                # only the last 30 days
llm spend --model gpt-5            # one model only
llm spend --json                   # machine-readable report
llm spend --prices my-prices.json  # override/extend the price table
llm spend --database /path/to/logs.db   # a non-default database
```

The database is opened **read-only** — the report never writes to,
migrates, or locks your live `logs.db`.

## How it works

- **Both storage eras.** Current `llm` versions log usage to the
  `turns` table; older versions used `responses`. A call is written to
  exactly one of them, so both are read and merged.
- **Cached tokens are priced at the cached rate** when the provider's
  token details record them (OpenAI- and Anthropic-style detail shapes
  are recognized) and the price table has a cached-input price. Anything
  unrecognized is priced at the full input rate — never
  double-discounted.
- **Unknown models are never silently $0.** A model with no price in the
  table is listed with its token counts and a `—` cost, and named in the
  notes. Calls that recorded no token usage at all are counted and
  excluded.
- **Model names** come from the logged `resolved_model` when present,
  so an alias you called is priced as the model that actually ran.

## Prices

The bundled table is **example data as of 2026-10-30** (GPT-5 family,
GPT-4o family, Claude Sonnet/Haiku 4.5, Gemini 2.5 Pro/Flash) — providers
change prices often, so verify before relying on totals, or supply your
own file (same shape as below; entries override or extend by name):

```json
{"models": {"my-model": {"input_per_m": 0.40, "output_per_m": 1.60,
                         "cached_input_per_m": 0.04}}}
```

Prices are USD per 1M tokens; `cached_input_per_m` is optional.

## Limitations

- **Estimates, not invoices.** The log records what the provider
  *reported*; free tiers, credits, batch discounts, and per-request
  fees are not in the log and cannot be priced from it.
- Embeddings usage is not priced in this version.
- Alias-only model ids that never resolved to a priced model stay
  unpriced until you add them via `--prices`.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
python -m pytest        # 11 tests, fully offline
```

Requires Python ≥ 3.10 and `llm` itself. Zero additional runtime
dependencies.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Gang Ji.

Not affiliated with the `llm` project; `llm` is by Simon Willison.
