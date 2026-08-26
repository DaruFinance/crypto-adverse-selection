# Locked plan v2: bounding the survivorship bias on the Bybit and Binance panels

This is a **wording revision** of `survivorship_bound.md`, which stays in place unchanged as
the record of what was frozen before the first delisted coin-day was measured. Nothing about
the design, the prediction, the coin set, the matched-date rule or the two weightings differs
between the two. The single change is in the paragraph on Bybit book time, restated as a
property of the panel's basis rather than as a history of the measurement code. Both files
carry their own row in `LOCK_HASHES.md`.

Study: *Adverse Selection Over-Consumes the Touch.* This file fixes the design of the
survivorship-bound measurement before any delisted coin-day is measured. Its SHA-256 is
recorded in `LOCK_HASHES.md` and is stamped from the commit that adds it.

The three shipped panels contain no delisted coin. Every coin in each panel trades on that
panel's last sampled date, so the universe behind the headline is a survivor set and the
paper currently argues the direction of that omission rather than measuring it. This plan
measures it.

## The prediction

Delisted perpetuals carry more adverse selection than surviving ones over the same dates,
so their pooled net entry markout is the more negative of the two, by an amount of order
0.2 to 1.0 bp at the 10-second horizon.

If that is right, the survivor panel understates how negative the true universe net is, and
the direction stated in the paper holds. If delisted coins come back no worse, or better,
the direction argument in the limitations section is false and comes out. Both outcomes are
reported. The result is written up whichever way it lands, and no coin is added to or
dropped from the frozen set below after the first measurement runs.

## Venues, and why not three

Bybit and Binance USD-M only. Both publish a retrievable per-symbol archive from which the
listing and delisting dates of the whole cross-section can be read, and Bybit additionally
publishes dated delisting announcements that corroborate the archive count.

Hyperliquid is out of scope. Both of its archive buckets are requester-pays and return 403
without credentials, so no listing or delisting record was obtainable, and the bound is
therefore scoped to the two centralized venues and stated as such wherever it appears.

## What was counted, and from what

Delisting counts come from each venue's own archive rather than from announcement pages: a
symbol is treated as delisted on the last date for which the venue publishes a trade file
for it. On Bybit this was corroborated against the venue's delisting announcements, which
name 109 distinct perpetual symbols inside the window against 110 found in the archive.

| source | URL | retrieved |
|---|---|---|
| Bybit trade archive index and per-symbol listings | `https://public.bybit.com/trading/` | 2026-08-25 |
| Bybit order-book archive probes | `https://quote-saver.bycsi.com/orderbook/linear/<SYM>/` | 2026-08-25 |
| Bybit delisting announcements | `https://api.bybit.com/v5/announcements/index?locale=en-US&type=delistings` | 2026-08-25 |
| Binance USD-M archive listing | `https://s3-ap-northeast-1.amazonaws.com/data.binance.vision` (prefix `data/futures/um/daily/`) | 2026-08-25 |

| venue | window | USDT perps listed at window start | delisted during window | death rate |
|---|---|---|---|---|
| Bybit | 2023-04-01 to 2025-08-18 | 206 | 30 | 14.6 percent |
| Binance USD-M | 2023-05-16 to 2024-03-29 | 185 | 4 | 2.2 percent |

Counting every USDT perpetual that traded at any point inside the window rather than only
those alive at its start gives 110 deaths among 667 on Bybit, 16.5 percent, and 5 among 271
on Binance. The listed-at-start figure is the one used for the bound, since it is the
denominator a study fixing its universe at the window's start would have faced.

Binance's low count is a property of its window and not of the venue: the `bookTicker` feed
the measurement needs exists only for the 320 days from 2023-05-16 to 2024-03-30, which is
the whole of that panel's window, so its cross-section has under eleven months in which to
die against Bybit's twenty-nine.

## The frozen coin set

Every delisted USDT perpetual with at least 30 trading days inside its venue's window
qualifies, of which 108 do on Bybit and 4 on Binance. Where more than 10 qualify the 10
with the most in-window trading days are taken, so the Bybit set below is the 10 longest-
lived delistings and not a sample of the 108.

**That selection rule has a known direction and it is recorded here rather than discovered
later.** The longest-lived delistings are the least distressed members of the dead set, so
the gap measured on them is expected to understate the gap across all 108, which makes the
resulting bound an understatement rather than a conservative overstatement. Any reading of
the number has to carry that.

**Bybit, 10 coins, 2,481 panel-matched coin-days**

| symbol | first in-window | last in-window | in-window days | on panel dates |
|---|---|---|---|---|
| BALUSDT | 2023-04-01 | 2025-07-29 | 851 | 285 |
| OMGUSDT | 2023-04-01 | 2025-06-03 | 795 | 265 |
| EOSUSDT | 2023-04-01 | 2025-05-21 | 782 | 261 |
| LEVERUSDT | 2023-06-05 | 2025-07-01 | 758 | 253 |
| MDTUSDT | 2023-06-30 | 2025-07-04 | 736 | 246 |
| LINAUSDT | 2023-04-01 | 2025-03-27 | 727 | 243 |
| BNXUSDT | 2023-04-01 | 2025-03-17 | 717 | 239 |
| RENUSDT | 2023-04-01 | 2025-03-06 | 706 | 236 |
| STMXUSDT | 2023-04-01 | 2025-02-23 | 695 | 232 |
| REEFUSDT | 2023-04-01 | 2025-01-22 | 663 | 221 |

**Binance USD-M, 4 coins, 970 coin-days.** All four qualifying delistings are taken.

| symbol | first in-window | last in-window | in-window days |
|---|---|---|---|
| BLUEBIRDUSDT | 2023-05-16 | 2024-03-26 | 316 |
| FOOTBALLUSDT | 2023-05-16 | 2024-03-26 | 316 |
| TOMOUSDT | 2023-05-16 | 2023-11-14 | 182 |
| STRAXUSDT | 2023-10-11 | 2024-03-15 | 156 |

BLUEBIRDUSDT and FOOTBALLUSDT are contracts on Binance index products rather than on a
single coin, so the Binance gap is reported twice, once on all four and once on TOMOUSDT
and STRAXUSDT alone, and the difference between the two is stated.

## The measurement

The existing decomposition, unchanged. Same posting rule, same last-in-queue fill rule,
same 100 ms re-quote gap, same 10-second and 60-second horizons, same venue maker fee, so
the rows are commensurable with the shipped panels by construction rather than by argument.

Two implementation points are fixed here because both were found to matter and neither is
free to choose later.

The Bybit book time is the venue's publish stamp `ts`, not its matching-engine stamp `cts`.
That is the basis the shipped Bybit panel is built on, and the delisted rows have to share it:
the two stamps differ by 3 ms at the median on BTC and 16 ms on a thinner symbol, and a
book-time shift of that size inside a dead-minus-live contrast would be indistinguishable from
the effect being measured. Every row records which basis produced it in its `ts_semantics`
field, and the commensurability check below is what establishes that the basis matches.

Commensurability is not asserted, it is checked: `reproduce/analysis/survivorship_bound.py`
re-measures shipped panel coin-days through the same code and requires every field to
reproduce the shipped value. That check passed on 16 of 16 Bybit coin-days spanning 2023 to
2025 and on the Binance coin-day tested, before this plan was locked.

## The comparison

Matched dates only. A delisted coin has data up to its death while survivors run to the end
of the window, so comparing the two raw would contrast 2023 against 2024 and 2025 rather
than dying against surviving.

For each delisted coin, its own in-window live dates are taken, intersected with the dates
its venue's shipped panel actually sampled, and the live-coin pooled net entry markout is
computed over exactly that date set. The contrast is the delisted coin's net minus the
matched-window live net, per coin, then pooled. On Bybit the panel samples a 3-day stride,
so intersecting with panel dates is what makes the two sides share a calendar exactly; on
Binance the panel is every calendar day and the intersection is the coin's own life.

The interval on the pooled gap is clustered two-way on month and coin, as everywhere else in
the paper, and the same Kish floor applies: below five effective clusters no interval is read
and no verdict is returned. Individual coins are expected to withhold, since a coin that
traded 40 days spans two months. The pooled gap is the number that carries and the per-coin
column is reported as dispersion.

## The bound

    bias ~ share of the universe that died  x  gap between dead and live

Computed twice and both reported, with the coin-count reading led on.

**By coin count.** The death rate above, 14.6 percent on Bybit and 2.2 percent on Binance,
multiplied by the per-coin gap. This treats every name as one unit, which is the pessimistic
reading.

**By fill weight.** The share of simulated fills the delisted coins carry over their matched
dates, multiplied by the gap. The headline is fill-count weighted and its weight sits in the
largest names, so this is expected to be much smaller than the coin-count reading. If it is,
that is reported as a finding in its own right: the pooled figure is weighted toward exactly
the names least exposed to the omission.

The fill-weight reading has a scope limit that is recorded now. Its numerator covers the 10
and 4 measured delistings, not all 108 and 4 that qualify, and its denominator is the shipped
panel's 13 and 15 survivors, not the venue's full cross-section. It is therefore a fill share
within the measured set and not within the venue, while the coin-count death rate is
universe-wide. The two are not interchangeable and neither is presented as the other.

The result is written as a bound and not as a correction: the survivorship gap moves the
pooled estimate by at most about X bp, in a stated direction. A correction would imply the
number can be subtracted to recover a survivorship-free estimate, which this does not
support, since the panel stays survivor-tilted and no point-in-time universe is constructed.

## What this does not do

It does not make any panel point-in-time. It does not close the Hyperliquid case. It does
not license any statement of the form "this is n percent of the venue's liquid universe",
which remains unsupported by the shipped artifacts.
