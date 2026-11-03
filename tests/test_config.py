import os
from pathlib import Path
import pytest
from research_agent.config import PREFIX, Settings
@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith(PREFIX):
            monkeypatch.delenv(name)


def test_defaults_when_nothing_is_set() -> None:
    settings = Settings.from_env()

    assert settings == Settings()
    assert settings.top_k == 3
    assert settings.retry_on_abstain is False


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVIDENCEGUARD_DATA_DIR", "my_docs")
    monkeypatch.setenv("EVIDENCEGUARD_TOP_K", "4")
    monkeypatch.setenv("EVIDENCEGUARD_RETRY_ON_ABSTAIN", "TRUE")
    monkeypatch.setenv("EVIDENCEGUARD_MIN_RETRY_SCORE", "0.5")

    settings = Settings.from_env()

    assert settings.data_dir == Path("my_docs")
    assert settings.top_k == 4
    assert settings.retry_on_abstain is True
    assert settings.min_retry_score == 0.5


def test_blank_values_fall_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVIDENCEGUARD_TOP_K", "   ")

    assert Settings.from_env().top_k == 3


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TOP_K", "three"),
        ("TOP_K", "0"),
        ("RETRY_ON_ABSTAIN", "maybe"),
        ("MIN_RETRY_SCORE", "high"),
        ("LLM_DTYPE", "float8"),
    ],
)
def test_invalid_values_raise_clear_errors(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(f"EVIDENCEGUARD_{name}", value)

    with pytest.raises(ValueError, match=name):
        Settings.from_env()