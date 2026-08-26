# Lock hashes

The plan file is frozen at the digest below. Editing it invalidates the hash. A revision
must be a new `-vN` sibling with its own row, never an in-place edit.

| File | SHA-256 | Stamped (UTC) |
|---|---|---|
| `survivorship_bound.md` | `a53b31606661adc36658f09f3d10ab5daaa6caef31236b32bcbb6abb064081e5` | 2026-08-25T18:08:25Z |
| `survivorship_bound-v2.md` | `757488631b80b96b1669fb485ca9b9284b59760f8ff4a854786066556ecd8ae7` | 2026-08-26T21:04:03Z |

Verify:

```bash
sha256sum reproduce/preregistration/survivorship_bound.md
```

## Ordering

The digest fixes the plan text. The ordering that matters for this measurement is that the
frozen coin set and the prediction precede the first delisted coin-day being measured, and
that is evidenced by the commit adding this directory: no row of
`reproduce/panels/delisted_coindays.csv` exists at that commit, and the panel is added by a
later one.

The counts and the coin set in the plan come from venue archive listings, which are reads of
public indexes and carry no measurement of the decomposition. The commensurability check
named in the plan re-measures coin-days that are already in the shipped panels and likewise
measures no delisted coin.
