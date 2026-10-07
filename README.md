# nfstream-adversary

## Usage examples

### Sweep `--nfstream-timeout` and run the default experiment on each

Extracts flows for `scenario1-1` at active_timeout values of 5s, 15s, 30s, and
1800s (nfstream's own default), each cached in its own `scenario1-1_timeout<N>`
folder, then runs `experiments/default.json` against each cached dataset.

```bash
for t in 5 15 30 1800; do
  python main.py --scenario scenario1-1 --nfstream-timeout "$t"
  python main.py experiment --scenario "scenario1-1_timeout${t}" --plan experiments/default.json
done
```

Each experiment's results land under `processed/scenario1-1_timeout<N>/results/experiments/default_experiment/`,
so results across timeouts can be compared without re-extraction clobbering each other.
