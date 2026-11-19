# GTA VI Trophy Rarity Forecaster

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tiars837/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tiars837/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts, for each of Grand Theft Auto VI's 55 PlayStation 5 trophies, the
earn rate on Exophase one week after launch (2026-11-26 00:00 UTC), as five
quantiles (5%, 25%, 50%, 75%, 95%). Entry for the make.fun bounty
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## Method

`main.ts` (Node 22, no dependencies, no network) reads every case file in
the input directory and writes one forecast per case. Per trophy it fits a
recency-weighted line of logit(earn rate) against log-time since release
over the last 36 hours of reads, reads the line at the target for the
median, and builds the spread from the fit's residual scale, the slope's
uncertainty over the extrapolation, and a random-walk term that grows with
the horizon. When a trophy's history is flat, it falls back to a dilution
prior: Exophase refreshes its counts in jumps, and on launch night one
refresh took the tracked base from 638 to 3,868 players and pushed almost
every rate down, so the median shifts down with a wide, honest band
instead of claiming the rate will sit still.

Run it locally:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

It is deterministic: two runs on the same case write identical bytes.

## License

MIT — see [LICENSE](LICENSE).
