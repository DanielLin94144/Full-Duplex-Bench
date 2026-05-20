#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
latency_injector.py

Configurable delay injection layer for mock APIs.
Simulates real-world API latency to test how voice agents handle
"system loading" time during multi-step tool calling.
"""

import time
import random
from dataclasses import dataclass, field
from typing import Optional


# ==============================================================================
# Latency Profiles
# ==============================================================================

@dataclass
class LatencyProfile:
    """Configuration for a single latency injection profile."""
    name: str
    min_ms: int = 0
    max_ms: int = 0
    fixed_ms: Optional[int] = None  # If set, overrides min/max
    progressive_factor: float = 1.0  # Multiplier applied per successive call
    jitter: bool = True  # Whether to add random jitter within range

    def get_delay_ms(self, call_index: int = 0) -> int:
        """Calculate the delay in ms for a given call index."""
        if self.fixed_ms is not None:
            base = self.fixed_ms
        elif self.jitter:
            base = random.randint(self.min_ms, self.max_ms)
        else:
            base = (self.min_ms + self.max_ms) // 2

        # Apply progressive slowdown
        if self.progressive_factor != 1.0 and call_index > 0:
            base = int(base * (self.progressive_factor ** call_index))

        return base


# Pre-defined latency profiles
LATENCY_PROFILES = {
    "instant": LatencyProfile(
        name="instant",
        fixed_ms=0,
        jitter=False,
    ),
    "fast": LatencyProfile(
        name="fast",
        min_ms=50,
        max_ms=200,
        jitter=True,
    ),
    "normal": LatencyProfile(
        name="normal",
        min_ms=200,
        max_ms=800,
        jitter=True,
    ),
    "slow": LatencyProfile(
        name="slow",
        min_ms=1000,
        max_ms=3000,
        jitter=True,
    ),
    "degraded": LatencyProfile(
        name="degraded",
        min_ms=500,
        max_ms=2000,
        progressive_factor=1.5,  # Gets 50% slower each successive call
        jitter=True,
    ),
    "timeout_risk": LatencyProfile(
        name="timeout_risk",
        min_ms=3000,
        max_ms=8000,
        jitter=True,
    ),
}


# Per-API default latency overrides (simulates different backend speeds)
API_LATENCY_DEFAULTS = {
    "search_flights":           "normal",
    "book_ticket":              "slow",      # Booking is slow
    "update_travel_profile":    "fast",
    "query_card_benefits":      "fast",
    "calculate_currency_exchange": "fast",
    "modify_autopay_source":    "slow",      # Financial mutations are slow
    "search_apartments":        "normal",
    "update_search_filter":     "fast",
    "calculate_commute":        "normal",
    "check_order_status":       "fast",
    "cancel_pending_action":    "normal",
    "process_exchange":         "slow",      # Processing is slow
}


# ==============================================================================
# Latency Injector
# ==============================================================================

@dataclass
class LatencyCallLog:
    """Log entry for a single latency-injected call."""
    api_name: str
    call_index: int
    profile_name: str
    injected_delay_ms: int
    timestamp: float


class LatencyInjector:
    """
    Wraps API calls with configurable latency injection.
    
    Usage:
        injector = LatencyInjector(profile="normal")
        
        # Option 1: Inject delay manually
        injector.inject("search_flights", call_index=0)
        result = search_flights(destination="Tokyo", date="2026-07-15")
        
        # Option 2: Use as decorator/wrapper
        delayed_search = injector.wrap(search_flights)
        result = delayed_search(destination="Tokyo", date="2026-07-15")
    """

    def __init__(self, profile="normal", per_api_profiles=None, enabled=True):
        """
        Args:
            profile: Default latency profile name (from LATENCY_PROFILES)
            per_api_profiles: Optional dict mapping api_name -> profile name
            enabled: Whether latency injection is active
        """
        self.default_profile = LATENCY_PROFILES.get(profile, LATENCY_PROFILES["normal"])
        self.per_api_profiles = per_api_profiles or {}
        self.enabled = enabled
        self.call_counts = {}  # Track call count per API for progressive delays
        self.call_log = []     # Full log of all injected delays

    def _get_profile(self, api_name):
        """Get the latency profile for a specific API."""
        if api_name in self.per_api_profiles:
            profile_name = self.per_api_profiles[api_name]
        elif api_name in API_LATENCY_DEFAULTS:
            profile_name = API_LATENCY_DEFAULTS[api_name]
        else:
            return self.default_profile

        return LATENCY_PROFILES.get(profile_name, self.default_profile)

    def inject(self, api_name, call_index=None):
        """
        Inject latency for a specific API call.
        
        Args:
            api_name: Name of the API being called
            call_index: Override call index for progressive delays.
                       If None, auto-increments per API.
        
        Returns:
            The delay in milliseconds that was injected
        """
        if not self.enabled:
            return 0

        if call_index is None:
            call_index = self.call_counts.get(api_name, 0)
            self.call_counts[api_name] = call_index + 1

        profile = self._get_profile(api_name)
        delay_ms = profile.get_delay_ms(call_index)

        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

        # Log
        log_entry = LatencyCallLog(
            api_name=api_name,
            call_index=call_index,
            profile_name=profile.name,
            injected_delay_ms=delay_ms,
            timestamp=time.time(),
        )
        self.call_log.append(log_entry)

        return delay_ms

    def wrap(self, func, api_name=None):
        """
        Wrap a function with latency injection.
        
        Args:
            func: The function to wrap
            api_name: API name override. Defaults to func.__name__
        
        Returns:
            Wrapped function with latency injection
        """
        name = api_name or func.__name__

        def wrapper(*args, **kwargs):
            delay = self.inject(name)
            result = func(*args, **kwargs)
            # Attach latency metadata to result if it's a dict
            if isinstance(result, dict):
                result["_latency_ms"] = delay
            return result

        wrapper.__name__ = f"latency_wrapped_{name}"
        wrapper.__doc__ = f"Latency-wrapped version of {name}"
        return wrapper

    def reset(self):
        """Reset call counts and logs."""
        self.call_counts.clear()
        self.call_log.clear()

    def get_stats(self):
        """Return summary statistics of injected latencies."""
        if not self.call_log:
            return {"total_calls": 0, "total_delay_ms": 0}

        delays = [e.injected_delay_ms for e in self.call_log]
        return {
            "total_calls": len(self.call_log),
            "total_delay_ms": sum(delays),
            "avg_delay_ms": round(sum(delays) / len(delays), 1),
            "max_delay_ms": max(delays),
            "min_delay_ms": min(delays),
            "per_api": {
                api: {
                    "calls": len([e for e in self.call_log if e.api_name == api]),
                    "avg_ms": round(
                        sum(e.injected_delay_ms for e in self.call_log if e.api_name == api)
                        / len([e for e in self.call_log if e.api_name == api]),
                        1,
                    ),
                }
                for api in set(e.api_name for e in self.call_log)
            },
        }

    def get_log(self):
        """Return the full call log as list of dicts."""
        return [
            {
                "api": e.api_name,
                "call_index": e.call_index,
                "profile": e.profile_name,
                "delay_ms": e.injected_delay_ms,
                "timestamp": e.timestamp,
            }
            for e in self.call_log
        ]


# ==============================================================================
# Main — Demo
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LATENCY INJECTOR — Demo")
    print("=" * 60)

    injector = LatencyInjector(profile="fast", enabled=True)

    # Simulate a multi-step tool calling sequence
    apis = ["search_flights", "book_ticket", "update_travel_profile"]
    for api in apis:
        delay = injector.inject(api)
        print(f"  {api}: injected {delay}ms delay")

    print(f"\n📊 Stats: {injector.get_stats()}")
