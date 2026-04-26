"""Mem0 memory plugin — MemoryProvider interface.

Server-side LLM fact extraction, semantic search with reranking, and
automatic deduplication via the Mem0 Platform API.

Original PR #2933 by kartik-mem0, adapted to MemoryProvider ABC.

Config via environment variables:
  MEM0_API_KEY       — Mem0 Platform API key (required)
  MEM0_USER_ID       — User identifier (default: hermes-user)
  MEM0_AGENT_ID      — Agent identifier (default: hermes)

Or via $HERMES_HOME/mem0.json.

Time-series awareness (added 2026-04-20):
  - Ebbinghaus decay formula: score(t) = (n_use)^β · e^(-λ·Δt) · s
  - Domain-aware half-life: volatile(3d) / normal(7d) / stable(30d)
  - Use count reinforcement: power-law weighting (β = 0.4-0.8)
  - Trust-weighted acceleration: κ = 2.0 for low-trust memories
  - Lifecycle quantization: 32→8→4→2 bit downgrade suggestions
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive failures, pause API calls
# for _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


# ---------------------------------------------------------------------------
# Time-Series Awareness — Ebbinghaus Decay (Industry Best Practices)
# ---------------------------------------------------------------------------
# References:
#   - CortexGraph: Ebbinghaus decay with power-law use reinforcement
#   - SuperLocalMemory V3.3: Multi-factor strength + quantization downgrade
#
# Core formula (CortexGraph):
#   score(t) = (n_use)^β · e^(-λ·Δt) · s
#   where λ = ln(2) / half_life

# _EBBINGHAUS_PARAMS is a frozen configuration — override at module level if needed.
_EBBINGHAUS_PARAMS = {
    "volatile": {  # Market data, real-time metrics
        "half_life_days": 3,
        "beta": 0.8,
        "forget_threshold": 0.10,
        "strength_default": 1.0,
    },
    "normal": {    # Projects, tools, general facts
        "half_life_days": 7,
        "beta": 0.6,
        "forget_threshold": 0.05,
        "strength_default": 1.0,
    },
    "stable": {    # User preferences, infrastructure
        "half_life_days": 30,
        "beta": 0.4,
        "forget_threshold": 0.02,
        "strength_default": 1.3,
    },
}

_TRUST_ACCELERATION_FACTOR = 2.0  # κ: low-trust memories decay faster

_DOMAIN_KEYWORDS = {
    "volatile": [
        r"nonfarm", r"gdp", r"spread", r"realtime", r"today", r"latest",
        r"stock price", r"exchange rate", r"interest rate", r"yield",
    ],
    "stable": [
        r"preference", r"server", r"graduated", r"degree", r"name", r"birthday",
        r"config", r"password", r"address", r"phone",
    ],
}

_LIFECYCLE_STATES = {
    "active":  {"retention": 0.8,  "bit_width": 32, "tag": ""},
    "warm":    {"retention": 0.5,  "bit_width": 8,  "tag": "⏳"},
    "cold":    {"retention": 0.2,  "bit_width": 4,  "tag": "⚠️"},
    "archive": {"retention": 0.05, "bit_width": 2,  "tag": "🔴"},
}


def _detect_domain(memory_text: str) -> str:
    """Detect domain type from memory content using keyword patterns."""
    if not memory_text:
        return "normal"
    text_lower = memory_text.lower()
    for kw in _DOMAIN_KEYWORDS["volatile"]:
        if re.search(kw, text_lower):
            return "volatile"
    for kw in _DOMAIN_KEYWORDS["stable"]:
        if re.search(kw, text_lower):
            return "stable"
    return "normal"


def _calculate_age_seconds(created_at: str) -> float:
    """Calculate age in seconds from an ISO-8601 timestamp.

    Returns 0.0 for unparseable inputs or future timestamps
    (negative age is meaningless for decay calculations).
    """
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - dt).total_seconds()
        return max(0.0, age)
    except Exception:
        return 0.0


def _calculate_ebbinghaus_score(
    age_seconds: float,
    n_use: int = 1,
    domain: str = "normal",
    strength: float | None = None,
    trust_weight: float = 1.0,
) -> float:
    """Ebbinghaus forgetting-curve score.

    Formula (CortexGraph):
        score(t) = (n_use)^β · e^(-λ·Δt) · s
    Enhanced with trust-weighted acceleration:
        λ_eff = λ · (1 + κ·(1 - trust))

    Half-life verification examples (n_use=1, trust_weight=1.0):
        >>> # Day 7, normal domain (half_life=7d, strength=1.0): score ≈ 0.500
        >>> round(_calculate_ebbinghaus_score(7 * 86400, domain="normal"), 3)
        0.5
        >>> # Day 3, volatile domain (half_life=3d, strength=1.0): score ≈ 0.500
        >>> round(_calculate_ebbinghaus_score(3 * 86400, domain="volatile"), 3)
        0.5
        >>> # Day 30, stable domain (half_life=30d, strength=1.3): score ≈ 0.650
        >>> round(_calculate_ebbinghaus_score(30 * 86400, domain="stable"), 3)
        0.65
    """
    params = _EBBINGHAUS_PARAMS.get(domain, _EBBINGHAUS_PARAMS["normal"])
    half_life_secs = params["half_life_days"] * 86400
    base_lambda = math.log(2) / half_life_secs
    effective_lambda = base_lambda * (1 + _TRUST_ACCELERATION_FACTOR * (1 - trust_weight))
    use_factor = math.pow(n_use, params["beta"])
    s = strength if strength is not None else params["strength_default"]
    score = use_factor * math.exp(-effective_lambda * age_seconds) * s
    return max(0.0, min(1.0, score))


def _determine_lifecycle(score: float, domain: str = "normal") -> dict:
    """Map retention score to lifecycle state + bit-width suggestion."""
    params = _EBBINGHAUS_PARAMS.get(domain, _EBBINGHAUS_PARAMS["normal"])
    if score > 0.8:
        state = "active"
    elif score > 0.5:
        state = "warm"
    elif score > 0.2:
        state = "cold"
    elif score > params["forget_threshold"]:
        state = "archive"
    else:
        state = "forgotten"
    lc = _LIFECYCLE_STATES.get(state, _LIFECYCLE_STATES["active"])
    return {
        "lifecycle_state": state,
        "retention_score": round(score, 3),
        "suggested_bit_width": lc["bit_width"],
        "freshness_tag": lc["tag"],
    }


def _add_time_aware_fields(memory: dict) -> dict:
    """Enrich a memory dict with Ebbinghaus time-series fields.

    Adds: age_days, domain, retention_score, lifecycle_state,
          suggested_bit_width, freshness_tag, n_use, eligible_for_promotion.
    """
    if not isinstance(memory, dict):
        return memory
    enhanced = dict(memory)
    created_at = memory.get("created_at", "")
    if not created_at:
        return enhanced

    age_secs = _calculate_age_seconds(created_at)
    age_days = int(age_secs / 86400)
    enhanced["age_days"] = age_days

    memory_text = memory.get("memory", memory.get("data", ""))
    domain = _detect_domain(memory_text)
    enhanced["domain"] = domain

    n_use = memory.get("access_count", memory.get("n_use", 1))
    enhanced["n_use"] = n_use

    params = _EBBINGHAUS_PARAMS.get(domain, _EBBINGHAUS_PARAMS["normal"])
    strength = memory.get("strength", params["strength_default"])
    trust_weight = 1.0
    meta = memory.get("metadata")
    if isinstance(meta, dict):
        trust_weight = meta.get("trust", 1.0)

    retention = _calculate_ebbinghaus_score(
        age_seconds=age_secs, n_use=n_use, domain=domain,
        strength=strength, trust_weight=trust_weight,
    )
    enhanced["retention_score"] = round(retention, 3)

    lc = _determine_lifecycle(retention, domain)
    enhanced["lifecycle_state"] = lc["lifecycle_state"]
    enhanced["suggested_bit_width"] = lc["suggested_bit_width"]
    enhanced["freshness_tag"] = lc["freshness_tag"]
    enhanced["eligible_for_promotion"] = n_use >= 5 and age_days <= 14

    return enhanced


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/mem0.json overrides.

    Environment variables provide defaults; mem0.json (if present) overrides
    individual keys.  This avoids a silent failure when the JSON file exists
    but is missing fields like ``api_key`` that the user set in ``.env``.
    """
    from hermes_constants import get_hermes_home

    config = {
        "api_key": os.environ.get("MEM0_API_KEY", ""),
        "user_id": os.environ.get("MEM0_USER_ID", "hermes-user"),
        "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"),
        "rerank": True,
        "keyword_search": False,
    }

    config_path = get_hermes_home() / "mem0.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PROFILE_SCHEMA = {
    "name": "mem0_profile",
    "description": (
        "Retrieve all stored memories about the user — preferences, facts, "
        "project context. Fast, no reranking. Use at conversation start."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search memories by meaning. Returns relevant facts ranked by similarity. "
        "Set rerank=true for higher accuracy on important queries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "rerank": {"type": "boolean", "description": "Enable reranking for precision (default: false)."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
        },
        "required": ["query"],
    },
}

CONCLUDE_SCHEMA = {
    "name": "mem0_conclude",
    "description": (
        "Store a durable fact about the user. Stored verbatim (no LLM extraction). "
        "Use for explicit preferences, corrections, or decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string", "description": "The fact to store."},
        },
        "required": ["conclusion"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class Mem0MemoryProvider(MemoryProvider):
    """Mem0 Platform memory with server-side extraction and semantic search."""

    def __init__(self):
        self._config = None
        self._client = None
        self._client_lock = threading.Lock()
        self._api_key = ""
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._rerank = True
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        cfg = _load_config()
        return bool(cfg.get("api_key"))

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/mem0.json."""
        import json
        from pathlib import Path
        config_path = Path(hermes_home) / "mem0.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    def get_config_schema(self):
        return [
            {"key": "api_key", "description": "Mem0 Platform API key", "secret": True, "required": True, "env_var": "MEM0_API_KEY", "url": "https://app.mem0.ai"},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {"key": "rerank", "description": "Enable reranking for recall", "default": "true", "choices": ["true", "false"]},
        ]

    def _get_client(self):
        """Thread-safe client accessor with lazy initialization."""
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                from mem0 import MemoryClient
                self._client = MemoryClient(api_key=self._api_key)
                return self._client
            except ImportError:
                raise RuntimeError("mem0 package not installed. Run: pip install mem0ai")

    def _is_breaker_open(self) -> bool:
        """Return True if the circuit breaker is tripped (too many failures)."""
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            # Cooldown expired — reset and allow a retry
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "Mem0 circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._api_key = self._config.get("api_key", "")
        # Prefer gateway-provided user_id for per-user memory scoping;
        # fall back to config/env default for CLI (single-user) sessions.
        self._user_id = kwargs.get("user_id") or self._config.get("user_id", "hermes-user")
        self._agent_id = self._config.get("agent_id", "hermes")
        self._rerank = self._config.get("rerank", True)

    def _read_filters(self) -> Dict[str, Any]:
        """Filters for search/get_all — scoped to user only for cross-session recall."""
        return {"user_id": self._user_id}

    def _write_filters(self) -> Dict[str, Any]:
        """Filters for add — scoped to user + agent for attribution."""
        return {"user_id": self._user_id, "agent_id": self._agent_id}

    @staticmethod
    def _unwrap_results(response: Any) -> list:
        """Normalize Mem0 API response — v2 wraps results in {"results": [...]}."""
        if isinstance(response, dict):
            return response.get("results", [])
        if isinstance(response, list):
            return response
        return []

    def system_prompt_block(self) -> str:
        return (
            "# Mem0 Memory\n"
            f"Active. User: {self._user_id}.\n"
            "Use mem0_search to find memories, mem0_conclude to store facts, "
            "mem0_profile for a full overview."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Mem0 Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _run():
            try:
                client = self._get_client()
                results = self._unwrap_results(client.search(
                    query=query,
                    filters=self._read_filters(),
                    rerank=self._rerank,
                    top_k=5,
                ))
                if results:
                    lines = [r.get("memory", "") for r in results if r.get("memory")]
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(f"- {l}" for l in lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Mem0 prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="mem0-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Send the turn to Mem0 for server-side fact extraction (non-blocking)."""
        if self._is_breaker_open():
            return

        def _sync():
            try:
                client = self._get_client()
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                client.add(messages, **self._write_filters())
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("Mem0 sync failed: %s", e)

        # Wait for any previous sync before starting a new one
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, CONCLUDE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "Mem0 API temporarily unavailable (multiple consecutive failures). Will retry automatically."
            })

        try:
            client = self._get_client()
        except Exception as e:
            return tool_error(str(e))

        if tool_name == "mem0_profile":
            try:
                memories = self._unwrap_results(client.get_all(filters=self._read_filters()))
                self._record_success()
                if not memories:
                    return json.dumps({"result": "No memories stored yet."})
                enhanced = [_add_time_aware_fields(m) for m in memories]
                lines = [m.get("memory", "") for m in enhanced if m.get("memory")]
                return json.dumps({"result": "\n".join(lines), "count": len(lines), "memories": enhanced})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to fetch profile: {e}")

        elif tool_name == "mem0_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            rerank = args.get("rerank", False)
            top_k = min(int(args.get("top_k", 10)), 50)
            try:
                results = self._unwrap_results(client.search(
                    query=query,
                    filters=self._read_filters(),
                    rerank=rerank,
                    top_k=top_k,
                ))
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = []
                for r in results:
                    e = _add_time_aware_fields(r)
                    items.append({
                        "memory": e.get("memory", ""),
                        "score": e.get("score", 0),
                        "age_days": e.get("age_days", 0),
                        "domain": e.get("domain", "normal"),
                        "retention_score": e.get("retention_score", 1.0),
                        "lifecycle_state": e.get("lifecycle_state", "active"),
                        "suggested_bit_width": e.get("suggested_bit_width", 32),
                        "freshness_tag": e.get("freshness_tag", ""),
                        "n_use": e.get("n_use", 1),
                        "eligible_for_promotion": e.get("eligible_for_promotion", False),
                    })
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Search failed: {e}")

        elif tool_name == "mem0_conclude":
            conclusion = args.get("conclusion", "")
            if not conclusion:
                return tool_error("Missing required parameter: conclusion")
            try:
                client.add(
                    [{"role": "user", "content": conclusion}],
                    **self._write_filters(),
                    infer=False,
                )
                self._record_success()
                return json.dumps({"result": "Fact stored."})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to store: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        with self._client_lock:
            self._client = None


def register(ctx) -> None:
    """Register Mem0 as a memory provider plugin."""
    ctx.register_memory_provider(Mem0MemoryProvider())
