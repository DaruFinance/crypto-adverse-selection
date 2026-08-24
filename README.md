# makercex

Measure the entry markout of a passive quote against true-aggressor-signed tape,
with cluster-robust intervals that abstain when the panel cannot support one.

Point it at your own book and tape:

```bash
pip install -e ".[data]"
makercex measure --snapshots book.parquet --trades tape.parquet --horizon 10s
```

```text
Entry markout at the touch, 10s horizon, pre-fee
  26,205 fills over 58 symbol-days (6 symbols, 10 months)

  captured half-spread     +0.5445 bp
  adverse selection        -0.8333 bp
  net entry markout        -0.2888 bp
  break-even rebate         0.2888 bp

  clustered on month and symbol, 10 clusters, 5.3 effective
  95% interval           [-0.7711, +0.1934]
  clears zero            no
```

Snapshots need a timestamp, the best bid and ask and the size at each. Trades
need a timestamp, price, size and a **true aggressor flag**. Column names are
detected from the usual spellings and can be overridden with `--*-col`;
timestamps may be integers in any unit or a datetime column; parquet and
delimited text both work. Each symbol-day is simulated separately and the
interval is taken across those cells, clustered on calendar month and symbol.

Below five effective clusters the tool returns no verdict rather than a weak
one. That is a withheld verdict, not a failed test, and the distinction is kept
everywhere in this package.

The aggressor flag is the one input the measurement cannot infer for you. A tick
rule or Lee-Ready in its place carries the error measured below.

## Main result

For a simulated last-in-queue touch quoter, the 10-second pre-fee entry markout
is negative on all three venues:

```text
Venue                         Capture  Adverse  Net       95% two-way CI
Bybit perpetuals                0.206   -1.044   -0.839   [-1.084, -0.594]
Binance USD-M perpetuals        0.460   -0.852   -0.391   [-0.452, -0.331]
Hyperliquid                     0.560   -1.017   -0.458   [-0.729, -0.186]
```

Clustering uses calendar month and coin. The three venue panels cover
different dates and coin sets, so the pooled values are not a causal venue
ranking.

Net is calculated as a one-sided entry markout:

```text
capture = maker_side * (mid_at_fill - fill_price) / mid_at_fill
adverse = maker_side * (mid_at_horizon - mid_at_fill) / mid_at_fill
net entry markout = capture + adverse
```

No inventory unwind or exit cost is modelled. The result therefore does not
establish realised maker profit or loss.

## What a trade-sign classifier costs

Almost every maker-markout study has to infer the aggressor side, and none can
measure the resulting error, because measuring it needs the true sign that was
missing. These venues publish it. The same decomposition, rerun on the same tape
under each rule:

```text
Venue        Rule            Accuracy  Capture  Adverse  Net      Error   Fills
Bybit        exchange flag      1.000    0.206   -1.044   -0.839       .  1.000
Bybit        Lee-Ready          0.932    0.216   -1.026   -0.810  +0.029  1.061
Bybit        tick rule          0.898    0.201   -0.997   -0.796  +0.043  0.937
Hyperliquid  exchange flag      1.000    0.560   -1.017   -0.458       .  1.000
Hyperliquid  Lee-Ready          0.918    0.533   -1.057   -0.524  -0.067  1.118
Hyperliquid  tick rule          0.894    0.547   -1.084   -0.537  -0.080  0.930
```

The sign of the result survives both classifiers on both venues. The magnitude
moves by 3 to 5 percent on Bybit and 15 to 17 percent on Hyperliquid, in
opposite directions, so no constant correction is available. One of the four
errors clears zero. A rule also changes which fills happen, not only how they
are signed, which is what the fills column is. Binance carries no trade archive
here and is the untested cell.

## Other measured results

**Matched venue days.** The same-date Bybit-minus-Hyperliquid estimate is
-0.355 bp at the selected 2,000 ms re-quote rung. Extraction choices affect the
estimate, which does not identify an architectural cause.

**Lead-lag.** The matched panel has a median 500 ms Bybit lead, while the median
publication-cadence gap is 441 ms. This panel cannot separate information
arrival from a publication-rate difference.

**Conditional results.** Bybit calm-to-stress sign tests return p-values of
0.789, 0.772 and 0.285 for the Asia, Europe and US session blocks. Under this
low-powered test, none of the blocks has a detectable difference.

**Depth sample.** In the 300-coin-day, bid-side Bybit depth panel, the minimum
measured break-even rebate is 0.328 bp, and the touch overtakes the deepest
quoted level at 0.771 bp, with levels referring to price depth rather than queue
position.

**Rebate qualification.** The published 1.5 bp tier carries obligations the
simulated quoter is not shown to satisfy.

**Quoting arm.** The tested Avellaneda-Stoikov depth rule remains negative on
66 of 91 coin-days and takes 9.2% as many fills as the touch rule. This single
arm does not represent inventory-aware market making in general.

**Multiplicity.** Across the 70 per-coin cells that return a verdict, 56 clear
zero at raw p below 0.05 and all 56 survive Benjamini-Hochberg; 54 survive
Benjamini-Yekutieli. Multiplicity is not what limits that surface.

The complete allowed and prohibited claim set is locked in
`reproduce/claim_contract.json`.

## The dataset

The panels have a name and a version so a result can be quoted against the bytes
it was computed on.

```text
MakerCEX Panels, version 1.0.0
  9 aggregate coin-day panels, 3 venues, 2023-04 to 2026-05
  9,983 coin-days, 125,828,922 simulated fills
  every panel's SHA-256 recorded in reproduce/lineage.json
```

Cite as: Gatto, D. V. (2026). *MakerCEX Panels, version 1.0.0.*
https://github.com/DaruFinance/crypto-adverse-selection

The version changes when a panel's bytes change. Run your own estimators against
`reproduce/panels/*.csv` directly; each row is a coin-day with fill counts and
the fill-weighted decomposition terms, which is everything the intervals in this
package are built from.

## Repository contents

```text
makercex/
  cli.py                    the `makercex measure` runner
  decomp.py                 entry-markout decomposition
  fills.py                  reference posting and fill rule
  inference.py              clustered estimators, abstention rule, BH and BY
  synth.py                  deterministic synthetic test fixtures
examples/
  run_synthetic.py          sub-minute mechanism and inference smoke test
reproduce/
  panels/                   nine aggregate analysis panels
  analysis/                 panel-to-result producers and checks
  build_panels.py           frozen-artifact-to-panel rebuild and hash check
  lineage.json              dataset identity, source and output hashes, boundary
  claim_contract.json       allowed and prohibited interpretations
  print_headline.py         compact result printer
  make_figures.py           figure producer
  *.json                    shipped analysis results
```

## Reproducing the papers

This repository is the reproducibility package for two working papers:

- *Adverse Selection Over-Consumes the Touch: A True-Aggressor-Signed
  Entry-Markout Decomposition Across Centralized and On-Chain Crypto Perpetual
  Order Books* — the measurement and its results.
- *Withholding a Verdict When the Panel Will Not Support One* — the interval
  calibration and the abstention rule, which are reusable on any panel.

Python 3.10 or newer is required. The smoke test finishes in under a minute,
asserting the invariants that cover the decomposition, its inference and the
fill rule.

```bash
pip install -e .
python examples/run_synthetic.py
python reproduce/build_panels.py --verify-shipped
python reproduce/print_headline.py
```

The same commands work in PowerShell:

```powershell
pip install -e .
python .\examples\run_synthetic.py
python .\reproduce\build_panels.py --verify-shipped
python .\reproduce\print_headline.py
```

Install the optional figure dependency and regenerate the figures with:

```bash
pip install -e ".[figures]"
python reproduce/make_figures.py
```

### Rebuild the analysis results

Run these from the repository root. Each producer writes to a separate file so
the shipped result remains untouched.

```bash
python reproduce/analysis/decomposition_table.py --out decomposition_by_venue.rebuilt.json
python reproduce/analysis/per_coin_intervals.py --out per_coin_intervals.rebuilt.json
python reproduce/analysis/per_coin_multiplicity.py --out per_coin_multiplicity.rebuilt.json
python reproduce/analysis/sign_rule_counterfactual.py --out sign_rule_counterfactual.rebuilt.json
python reproduce/analysis/conditional_net.py --block asia --out conditional_net_asia.rebuilt.json
python reproduce/analysis/conditional_net.py --block europe --out conditional_net_europe.rebuilt.json
python reproduce/analysis/conditional_net.py --block us --out conditional_net_us.rebuilt.json
python reproduce/analysis/depth_rebate_frontier.py --out depth_rebate_frontier.rebuilt.json
python reproduce/analysis/cross_venue_leadlag.py --out cross_venue_leadlag.rebuilt.json
python reproduce/analysis/depth_rebate_intervals.py --out depth_rebate_intervals.rebuilt.json
python reproduce/analysis/quoting_arm.py --out avellaneda_stoikov_arm.rebuilt.json
python reproduce/analysis/venue_difference.py --out venue_difference.rebuilt.json
```

Use `reproduce/analysis/compare_rebuild.py` to compare a rebuilt result with its
shipped counterpart. The producer outputs focus on published measurements and
may omit stored metadata or row-level diagnostic blocks.

The interval coverage study is intentionally separate because it takes longer:

```bash
python reproduce/analysis/interval_coverage.py --out coverage.rebuilt.json
python reproduce/analysis/wild_test_calibration.py --out wild_test_calibration.rebuilt.json
```

## Panel lineage and data boundary

All nine shipped panels rebuild byte-for-byte from frozen derived measurement
artifacts. `reproduce/lineage.json` records every source path relative to an
external artifact root, its SHA-256 hash, each output hash and each row count.

Verify the panels already in the repository with:

```bash
python reproduce/build_panels.py --verify-shipped
```

Rebuild them from a local copy of the frozen artifacts with:

```bash
python reproduce/build_panels.py \
  --source-root /path/to/measurement-artifacts \
  --out /path/to/rebuilt-panels
```

Artifact layout and hashes are defined in the manifest. A mismatched source is
rejected before extraction, and every rebuilt output is then hash-checked.

Here is the reproducibility boundary:

```text
raw venue archives
  -> venue-specific measurement programs and fill ledgers, not shipped
  -> frozen derived measurement artifacts, named and hashed in lineage.json
  -> aggregate panels, rebuilt by build_panels.py
  -> result JSON, tables and figures, rebuilt by reproduce/analysis
```

Size and data-license constraints keep raw archives and fill-level ledgers out
of the repository. The reproducible chain starts at the frozen artifacts and
ends at the figures; raw-archive processing remains outside it. Binance has the
same boundary as the other venues, with no shipped or independently rebuilt
Binance fill ledger.

That boundary is about *these* raw archives, which are not ours to
redistribute. It says nothing about yours, which is why `makercex measure` ships
even though the archives do not.

## Measurement rule

At each snapshot, the reference quoter posts one unit at the best bid and one
unit at the best ask behind all displayed size while clearing previous priority.
A fill requires an opposite-side aggressor at the quoted price whose
traded size exceeds the displayed queue ahead, and only the excess fills the
simulated order up to its remaining size.

Orders normally live for one snapshot interval, except that the final snapshot's
order persists until the tape ends because no later snapshot retires it. Mid
prices use the last observation at or before each requested timestamp without
interpolation. Side comes from true aggressor flags rather than a tick or quote
rule.

`makercex/fills.py` is the executable reference. The aggregate panels were
created by venue-specific measurement programs outside this package, so the
reference implementation documents and tests the rule but does not claim to
recreate the raw venue processing inside this repository.

## Inference and interpretation

The headline interval uses two-way month and coin clustering. Per-coin
intervals cluster by calendar month. Auxiliary analyses use the unit appropriate
to their panel, including dates, coin-days or coins.

Interpretation is limited in six ways.

**Study type.** This fixed-quoter measurement has no signal selection, portfolio
assembly or walk-forward strategy claim.

**Ratio.** The adverse-to-capture ratio becomes unstable as captured spread
approaches zero, making its 1.8 to 5.1 range unsuitable as a headline effect.

**Profitability.** Entry markout leaves out rebates, inventory management, exits
and other revenue, so it cannot establish whether market makers lose money.

**Venue comparison.** Different calendars, extraction choices and publication
cadence prevent causal architecture or information-lead claims.

**Conditional contrast.** A failure to reject does not demonstrate that the
result is unchanged across regimes.

**Depth.** Levels count prices away from the mid rather than positions within a
queue.

`reproduce/claim_contract.json` is the machine-readable interpretation boundary.
`reproduce/analysis/readme_claims.py` checks the numerical statements above
against the shipped result files and rejects prohibited phrases.

## Citation

```bibtex
@article{gatto2026adverse,
  author = {Gatto, Daniel V.},
  title = {Adverse Selection Over-Consumes the Touch: A True-Aggressor-Signed
           Entry-Markout Decomposition Across Centralized and On-Chain Crypto
           Perpetual Order Books},
  year = {2026}
}

@article{gatto2026withholding,
  author = {Gatto, Daniel V.},
  title = {Withholding a Verdict When the Panel Will Not Support One:
           Abstention, and the Joint Failure Region of Four Cluster-Robust
           Interval Methods on Small Unequally Weighted Panels},
  year = {2026}
}

@misc{makercexpanels,
  author = {Gatto, Daniel V.},
  title  = {MakerCEX Panels},
  year   = {2026},
  note   = {Version 1.0.0},
  url    = {https://github.com/DaruFinance/crypto-adverse-selection}
}
```

## License

Code is released under the MIT License. The aggregate panels are provided for
reproducibility subject to the original venues' data terms.
