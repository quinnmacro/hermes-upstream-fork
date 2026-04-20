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
# Time-Series Awareness Helper Functions (Industry Best Practices)
# ---------------------------------------------------------------------------
# References:
# - CortexGraph: Ebbinghaus decay with power-law use reinforcement
# - SuperLocalMemory V3.3: Multi-factor strength + quantization downgrade
# 
# Core formula (CortexGraph):
#   score(t) = (n_use)^β · e^(-λ·Δt) · s
#   where λ = ln(2) / half_life

# Ebbinghaus decay parameters per domain
EBBINGHAUS_PARAMS = {
    "volatile": {  # Market data, real-time metrics
        "half_life_days": 3,
        "beta": 0.8,
        "forget_threshold": 0.10,
        "strength_default": 1.0,
    },
    "normal": {    # Projects, tools
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

# Trust-weighted decay acceleration (SuperLocalMemory V3.3)
TRUST_ACCELERATION_FACTOR = 2.0  # κ: low-trust memories decay 3× faster

# Domain detection keywords
DOMAIN_KEYWORDS = {
    "volatile": [
        r"非农", r"GDP", r"利差", r"实时", r"今日", r"最新",
        r"股价", r"汇率", r"利率", r"收益率", r"spread",
    ],
    "stable": [
        r"偏好", r"服务器", r"毕业", r"学位", r"姓名", r"生日",
        r"配置", r"密码", r"地址", r"电话",
    ],
}

# Lifecycle quantization downgrade mapping (SuperLocalMemory V3.3)
LIFECYCLE_STATES = {
    "active": {"retention": 0.8, "bit_width": 32, "tag": ""},
    "warm": {"retention": 0.5, "bit_width": 8, "tag": "⏳"},
    "cold": {"retention": 0.2, "bit_width": 4, "tag": "⚠️"},
    "archive": {"retention": 0.05, "bit_width": 2, "tag": "🔴"},
}


def _detect_domain(memory_text: str) -> str:
    """Detect domain type from memory content using regex patterns."""
    text_lower = memory_text.lower()
    
    for keyword in DOMAIN_KEYWORDS["volatile"]:
        if re.search(keyword, text_lower):
            return "volatile"
    
    for keyword in DOMAIN_KEYWORDS["stable"]:
        if re.search(keyword, text_lower):
            return "stable"
    
    return "normal"


def _calculate_age_seconds(created_at: str) -> float:
    """Calculate age in seconds from ISO timestamp (precision for decay)."""
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return (now - dt).total_seconds()
    except Exception:
        return 0.0


def _calculate_age_days(created_at: str) -> int:
    """Calculate age in days from ISO timestamp."""
    seconds = _calculate_age_seconds(created_at)
    return int(seconds / 86400)


def _calculate_ebbinghaus_score(
    age_seconds: float,
    n_use: int = 1,
    domain: str = "normal",
    strength: float = None,
    trust_weight: float = 1.0,
) -> float:
    """
    Calculate Ebbinghaus forgetting curve score.
    
    Formula (CortexGraph):
        score(t) = (n_use)^β · e^(-λ·Δt) · s
        
    Enhanced with trust-weighted acceleration:
        λ_eff = λ · (1 + κ·(1 - trust))
    
    Args:
        age_seconds: Time since creation (seconds)
        n_use: Number of times memory was accessed (default 1)
        domain: Domain type for half-life tuning
        strength: Memory strength parameter (0-2)
        trust_weight: Trust score (0-1), low trust accelerates decay
    
    Returns:
        score: Retention score (0-1), lower = more forgotten
    """
    params = EBBINGHAUS_PARAMS.get(domain, EBBINGHAUS_PARAMS["normal"])
    
    # Decay constant λ = ln(2) / half_life
    half_life_seconds = params["half_life_days"] * 86400
    base_lambda = math.log(2) / half_life_seconds
    
    # Trust-weighted acceleration (κ = 2.0)
    kappa = TRUST_ACCELERATION_FACTOR
    effective_lambda = base_lambda * (1 + kappa * (1 - trust_weight))
    
    # Use count reinforcement (power-law)
    beta = params["beta"]
    use_factor = math.pow(n_use, beta)
    
    # Strength parameter
    s = strength or params["strength_default"]
    
    # Exponential decay
    decay_factor = math.exp(-effective_lambda * age_seconds)
    
    # Final score
    score = use_factor * decay_factor * s
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


def _determine_lifecycle_state(score: float, domain: str = "normal") -> Dict[str, Any]:
    """
    Determine lifecycle state based on retention score.
    
    Mapping (SuperLocalMemory V3.3):
        Active:   R > 0.8  → 32-bit
        Warm:     0.5 < R ≤ 0.8 → 8-bit
        Cold:     0.2 < R ≤ 0.5 → 4-bit
        Archive:  0.05 < R ≤ 0.2 → 2-bit
        Forgotten: R ≤ 0.05 → deleted
    """
    params = EBBINGHAUS_PARAMS.get(domain, EBBINGHAUS_PARAMS["normal"])
    
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
    
    lifecycle = LIFECYCLE_STATES.get(state, LIFECYCLE_STATES["active"])
    
    return {
        "lifecycle_state": state,
        "retention_score": round(score, 3),
        "suggested_bit_width": lifecycle["bit_width"],
        "freshness_tag": lifecycle["tag"],
    }


def _add_time_aware_fields(memory: dict) -> dict:
    """
    Add time-series awareness fields to a memory object.
    
    Fields added:
        - age_days: Days since creation
        - domain: Detected domain type (volatile/normal/stable)
        - retention_score: Ebbinghaus decay score (0-1)
        - lifecycle_state: active/warm/cold/archive/forgotten
        - suggested_bit_width: Quantization downgrade suggestion (32/8/4/2)
        - freshness_tag: Visual indicator
        - n_use: Use count (if available, default 1)
        - eligible_for_promotion: True if n_use ≥ 5 and age ≤ 14 days
    """
    if not isinstance(memory, dict):
        return memory
    
    enhanced = dict(memory)
    
    created_at = memory.get("created_at", "")
    if not created_at:
        return enhanced
    
    # Age calculation
    age_seconds = _calculate_age_seconds(created_at)
    age_days = int(age_seconds / 86400)
    enhanced["age_days"] = age_days
    
    # Domain detection
    memory_text = memory.get("memory", memory.get("data", ""))
    domain = _detect_domain(memory_text)
    enhanced["domain"] = domain
    
    # Use count (default 1 if not tracked)
    n_use = memory.get("access_count", memory.get("n_use", 1))
    enhanced["n_use"] = n_use
    
    # Strength (default from domain params)
    params = EBBINGHAUS_PARAMS.get(domain, EBBINGHAUS_PARAMS["normal"])
    strength = memory.get("strength", params["strength_default"])
    
    # Trust weight (metadata-derived or default)
    trust_weight = 1.0
    if memory.get("metadata") and isinstance(memory["metadata"], dict):
        trust_weight = memory["metadata"].get("trust", 1.0)
    
    # Ebbinghaus score
    retention_score = _calculate_ebbinghaus_score(
        age_seconds=age_seconds,
        n_use=n_use,
        domain=domain,
        strength=strength,
        trust_weight=trust_weight,
    )
    enhanced["retention_score"] = round(retention_score, 3)
    
    # Lifecycle state
    lifecycle = _determine_lifecycle_state(retention_score, domain)
    enhanced["lifecycle_state"] = lifecycle["lifecycle_state"]
    enhanced["suggested_bit_width"] = lifecycle["suggested_bit_width"]
    enhanced["freshness_tag"] = lifecycle["freshness_tag"]
    
    # Promotion eligibility (CortexGraph rule)
    if n_use >= 5 and age_days <= 14:
        enhanced["eligible_for_promotion"] = True
    else:
        enhanced["eligible_for_promotion"] = False
    
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
    }

    config_path = os.path.join(get_hermes_home(), "mem0.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                overrides = json.load(f)
            if isinstance(overrides, dict):
                config.update(overrides)
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to read %s; using env defaults", config_path)

    return config


class Mem0MemoryProvider(MemoryProvider):
    """MemoryProvider backed by Mem0 Platform API (api.mem0.ai)."""

    def __init__(self):
        self._config = {}
        self._api_key = ""
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._rerank = True
        self._client = None
        self._client_lock = threading.Lock()
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._prefetch_result = ""
        self._sync_thread = None
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    def _get_client(self):
        """Lazy-init mem0ai client; thread-safe."""
        if self._client is not None:
            return self._client

        with self._client_lock:
            if self._client is not None:
                return self._client

            if not self._api_key:
                raise ValueError("MEM0_API_KEY not set")

            from mem0.client import MemoryClient
            self._client = MemoryClient(api_key=self._api_key)
            return self._client

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.time() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "Mem0 API failed %d consecutive times; pausing for %ds",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    def _is_breaker_open(self) -> bool:
        if time.time() < self._breaker_open_until:
            return True
        return False

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
            f"Active. User: {self._user_id}.\\n"
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
                        self._prefetch_result = "\\n".join(f"- {l}" for l in lines)
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
                
                # Add time-series awareness fields
                enhanced_memories = [_add_time_aware_fields(m) for m in memories]
                
                lines = [m.get("memory", "") for m in enhanced_memories if m.get("memory")]
                return json.dumps({
                    "result": "\\n".join(lines),
                    "count": len(lines),
                    "memories": enhanced_memories,  # Full data with time fields
                })
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
                
                # Add time-series awareness fields to each result
                items = []
                for r in results:
                    enhanced = _add_time_aware_fields(r)
                    items.append({
                        "memory": enhanced.get("memory", ""),
                        "score": enhanced.get("score", 0),
                        "age_days": enhanced.get("age_days", 0),
                        "domain": enhanced.get("domain", "normal"),
                        "retention_score": enhanced.get("retention_score", 1.0),
                        "lifecycle_state": enhanced.get("lifecycle_state", "active"),
                        "suggested_bit_width": enhanced.get("suggested_bit_width", 32),
                        "freshness_tag": enhanced.get("freshness_tag", ""),
                        "n_use": enhanced.get("n_use", 1),
                        "eligible_for_promotion": enhanced.get("eligible_for_promotion", False),
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


# Tool schemas (unchanged from original)
PROFILE_SCHEMA = {
    "name": "mem0_profile",
    "description": "Retrieve all stored memories about the user",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": "Search stored memories for specific information",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
            "rerank": {
                "type": "boolean",
                "description": "Enable reranking for better relevance",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default 10, max 50)",
            },
        },
        "required": ["query"],
    },
}

CONCLUDE_SCHEMA = {
    "name": "mem0_conclude",
    "description": "Store a durable fact about the user",
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {
                "type": "string",
                "description": "The fact to store",
            },
        },
        "required": ["conclusion"],
    },
}