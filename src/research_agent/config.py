import os
from dataclasses import dataclass
from pathlib import Path

from research_agent.generator import DEFAULT_DTYPE, DEFAULT_LLM_NAME, SUPPORTED_DTYPES

PREFIX = "EVIDENCEGUARD_"
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _raw(name: str) -> str | None:
    value = os.environ.get(PREFIX + name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _read_int(name: str, default: int, minimum: int = 1) -> int:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{PREFIX}{name} must be an integer, got {raw!r}.") from None
    if value < minimum:
        raise ValueError(f"{PREFIX}{name} must be at least {minimum}, got {value}.")
    return value


def _read_float(name: str, default: float) -> float:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{PREFIX}{name} must be a number, got {raw!r}.") from None


def _read_bool(name: str, default: bool) -> bool:
    raw = _raw(name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(f"{PREFIX}{name} must be true or false, got {raw!r}.")


@dataclass(frozen=True)
class Settings:
    """Everything the application can be configured with."""

    data_dir: Path = Path("data/raw")
    db_path: Path = Path("chroma_db/app")
    llm_name: str = DEFAULT_LLM_NAME
    llm_dtype: str = DEFAULT_DTYPE
    top_k: int = 3
    widen_k: int = 5
    max_attempts: int = 3
    max_new_tokens: int = 150
    retry_on_abstain: bool = False
    min_retry_score: float = 0.35  # not tuned: a starting value, see README

    @classmethod
    def from_env(cls) -> "Settings":
        """Read settings from EVIDENCEGUARD_* variables, falling back to defaults."""
        defaults = cls()
        llm_dtype = _raw("LLM_DTYPE") or defaults.llm_dtype
        if llm_dtype not in SUPPORTED_DTYPES:
            raise ValueError(
                f"{PREFIX}LLM_DTYPE must be one of {sorted(SUPPORTED_DTYPES)}, got {llm_dtype!r}."
            )
        return cls(
            data_dir=Path(_raw("DATA_DIR") or defaults.data_dir),
            db_path=Path(_raw("DB_PATH") or defaults.db_path),
            llm_name=_raw("LLM_NAME") or defaults.llm_name,
            llm_dtype=llm_dtype,
            top_k=_read_int("TOP_K", defaults.top_k),
            widen_k=_read_int("WIDEN_K", defaults.widen_k),
            max_attempts=_read_int("MAX_ATTEMPTS", defaults.max_attempts),
            max_new_tokens=_read_int("MAX_NEW_TOKENS", defaults.max_new_tokens),
            retry_on_abstain=_read_bool("RETRY_ON_ABSTAIN", defaults.retry_on_abstain),
            min_retry_score=_read_float("MIN_RETRY_SCORE", defaults.min_retry_score),
        )