# Changelog

## 0.1.0 — 2026-10-30

- Initial release: `llm spend` prices the token usage already logged in
  `llm`'s logs.db and reports estimated spend grouped by model
  (default) or by day, with `--days`, `--model`, `--prices`, and
  `--json` options.
- Reads both storage eras — the current `turns` table and the legacy
  `responses` table — merged, always read-only.
- Cached input tokens are priced at the cached-input rate when the
  logged token details record them (OpenAI- and Anthropic-style
  shapes); models without a price are reported with token counts and
  an unknown cost rather than $0, and calls that recorded no usage are
  counted and excluded.
- Bundled example price table (as of 2026-10-30) covering the GPT-5
  and GPT-4o families, Claude Sonnet/Haiku 4.5, and Gemini 2.5
  Pro/Flash.
