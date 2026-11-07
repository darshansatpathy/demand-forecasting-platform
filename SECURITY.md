# Security policy

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's
[private vulnerability reporting](https://github.com/darshansatpathy/demand-forecasting-platform/security/advisories/new)
for this repository. You should get an acknowledgement within three working days.

## Operating it safely

- The API has no authentication; put it behind your gateway.
- Forecast tables are Parquet; the model is LightGBM's text format. Nothing is unpickled.
- If you connect a language model to `insights.narrate`, it receives the answer text only, never
  the underlying tables.
