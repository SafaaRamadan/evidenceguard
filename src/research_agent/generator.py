"""Run a small local chat model to generate text."""

import logging
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DTYPE = "bfloat16"
SUPPORTED_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


@dataclass(frozen=True)
class GenerationResult:
    """The generated text together with how it was produced."""

    text: str
    prompt_tokens: int
    new_tokens: int
    seconds: float

    @property
    def tokens_per_second(self) -> float:
        return self.new_tokens / self.seconds if self.seconds > 0 else 0.0


class Generator:
    """Wraps one local chat model. Create it once, then reuse it."""

    def __init__(
        self, model_name: str = DEFAULT_LLM_NAME, dtype_name: str = DEFAULT_DTYPE
    ) -> None:
        if dtype_name not in SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported dtype '{dtype_name}'. Choose from {sorted(SUPPORTED_DTYPES)}."
            )
        start = time.perf_counter()
        self.model_name = model_name
        self.dtype_name = dtype_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=SUPPORTED_DTYPES[dtype_name]
        )
        self._model.eval()
        logger.info(
            "Loaded %s (%s) in %.1f s", model_name, dtype_name, time.perf_counter() - start
        )

    def generate(
        self, messages: list[dict[str, str]], max_new_tokens: int = 200
    ) -> GenerationResult:
        """Answer a chat. Greedy decoding, so the same input gives the same output."""
        if not messages:
            raise ValueError("messages must not be empty.")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1.")

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        prompt_tokens = inputs["input_ids"].shape[1]

        start = time.perf_counter()
        with torch.inference_mode():
            output = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        seconds = time.perf_counter() - start

        new_ids = output[0][prompt_tokens:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            new_tokens=len(new_ids),
            seconds=seconds,
        )


if __name__ == "__main__":
    import sys

    import psutil

    logging.basicConfig(level=logging.INFO)
    for noisy in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    def peak_memory_mb() -> float:
        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss) / 1024**2

    generator = Generator(dtype_name=sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DTYPE)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "In one sentence, what is retrieval-augmented generation?"},
    ]

    generator.generate(messages, max_new_tokens=10)  # warm-up, not timed
    for run in range(1, 4):
        result = generator.generate(messages, max_new_tokens=60)
        print(
            f"run {run}: {result.new_tokens} tokens in {result.seconds:.1f} s "
            f"({result.tokens_per_second:.1f} tokens/s)"
        )
    print(f"dtype: {generator.dtype_name}, peak memory: {peak_memory_mb():.0f} MB")
    print("Answer:", result.text)