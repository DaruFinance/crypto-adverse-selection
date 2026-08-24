"""
Entry-markout decomposition and cluster-robust inference for crypto perpetuals.
"""

from .decomp import (adverse_selection_bp, base_symbol, breakeven_rebate_bp,
                     decompose, net_after_fee, spread_capture_bp)
from .fills import (DUST_FRAC, markout_mids, mid_at_or_before,
                    simulate_touch_fills)
from .inference import (MIN_CLUSTERS, benjamini_hochberg, cluster_bootstrap,
                        effective_clusters, sign_test_p, t_crit_95,
                        t_two_sided_p, two_way_se, wild_cluster_p)
from .synth import make_panel, make_tape

__all__ = [
    "base_symbol", "simulate_touch_fills", "DUST_FRAC",
    "mid_at_or_before", "markout_mids",
    "spread_capture_bp", "adverse_selection_bp", "decompose", "net_after_fee",
    "breakeven_rebate_bp", "cluster_bootstrap", "two_way_se", "wild_cluster_p",
    "effective_clusters", "sign_test_p", "make_tape", "make_panel",
    "t_crit_95", "MIN_CLUSTERS", "t_two_sided_p", "benjamini_hochberg",
]
__version__ = "1.0.0"
