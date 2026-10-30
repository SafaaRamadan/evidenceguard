import pytest

from research_agent.generator import Generator

pytestmark = pytest.mark.slow

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say hello."},
]


@pytest.fixture(scope="module")
def generator() -> Generator:
    return Generator()


def test_generate_returns_text_and_measurements(generator: Generator) -> None:
    result = generator.generate(MESSAGES, max_new_tokens=20)

    assert result.text != ""
    assert result.prompt_tokens > 0
    assert 0 < result.new_tokens <= 20
    assert result.seconds > 0
    assert result.tokens_per_second > 0


def test_max_new_tokens_is_respected(generator: Generator) -> None:
    result = generator.generate(MESSAGES, max_new_tokens=5)

    assert result.new_tokens <= 5


def test_greedy_decoding_is_deterministic(generator: Generator) -> None:
    first = generator.generate(MESSAGES, max_new_tokens=20)
    second = generator.generate(MESSAGES, max_new_tokens=20)

    assert first.text == second.text


def test_empty_messages_raise(generator: Generator) -> None:
    with pytest.raises(ValueError):
        generator.generate([])


def test_invalid_max_new_tokens_raises(generator: Generator) -> None:
    with pytest.raises(ValueError):
        generator.generate(MESSAGES, max_new_tokens=0)


def test_unsupported_dtype_raises_before_loading() -> None:
    with pytest.raises(ValueError):
        Generator(dtype_name="float8")