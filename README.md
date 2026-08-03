# Crypto adverse selection at the touch

This repository is the reproducibility package for *Adverse Selection
Over-Consumes the Touch: A True-Aggressor-Signed Maker P&L Decomposition Across
Centralized and On-Chain Crypto Perpetual Order Books*.

It contains the reference measurement code, aggregate coin-day panels, analysis
scripts and figure producers. It does not contain raw exchange data.

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

The complete allowed and prohibited claim set is locked in
`reproduce/claim_contract.json`.

## Repository contents

```text
makercex/
  decomp.py                 entry-markout decomposition
  fills.py                  reference posting and fill rule
  inference.py              clustered estimators and sign tests
  synth.py                  deterministic synthetic test fixtures
examples/
  run_synthetic.py          sub-minute mechanism and inference smoke test
reproduce/
  panels/                   eight aggregate analysis panels
  analysis/                 panel-to-result producers and checks
  build_panels.py           frozen-artifact-to-panel rebuild and hash check
  lineage.json              source and output hashes, row counts and boundary
  claim_contract.json       allowed and prohibited interpretations
  print_headline.py         compact result printer
  make_figures.py           figure producer
  *.json                    shipped analysis results
```

## Quick start

Python 3.10 or newer is required. The smoke test finishes in under a minute,
asserting 53 invariants that cover the decomposition and its inference and fill
rule.

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

## Rebuild the analysis results

Run these commands from the repository root. Each producer writes to a separate
file so the shipped result remains untouched.

```bash
python reproduce/analysis/decomposition_table.py --out decomposition_by_venue.rebuilt.json
python reproduce/analysis/per_coin_intervals.py --out per_coin_intervals.rebuilt.json
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

All eight shipped panels rebuild byte-for-byte from frozen derived measurement
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
           Maker P\&L Decomposition Across Centralized and On-Chain Crypto
           Perpetual Order Books},
  year = {2026}
}
```

## License

Code is released under the MIT License. The aggregate panels are provided for
reproducibility subject to the original venues' data terms.
