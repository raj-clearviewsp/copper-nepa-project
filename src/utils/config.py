"""Application-wide configuration and constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class NepaConfig:
    """Configuration switches for NEPA analytics business rules."""

    eis_cap_days: int = 730
    ea_cap_days: int = 365
    overlap_allocation: str = "equal"  # equal | primary | proportional
    compressible_steps: List[str] = field(default_factory=lambda: ["B", "D"])
    litigation_scale: float = 1.0
    rework_scale: float = 1.0
    policy_change_scale: float = 1.0
    reuse_scale: float = 1.0
    start_cutoff_date: str | None = None
    outlier_percentile_cap: float = 0.99
    include_active_actions: bool = False
    include_flagged: bool = False
    commodity: str = "Copper"

    @property
    def cap_lookup(self) -> Dict[str, int]:
        return {
            "EIS": self.eis_cap_days,
            "EA": self.ea_cap_days,
        }


DEFAULT_CONFIG = NepaConfig()
