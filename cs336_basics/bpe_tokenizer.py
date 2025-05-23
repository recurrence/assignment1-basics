from collections.abc import Iterable, Iterator
from functools import reduce
import pickle
import sys

from cs336_basics.pre_tokenizer import PreTokenizer


class BPETokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.inverted_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_dict = {tuple(k): i for i, k in enumerate(merges)}
        self.special_tokens = special_tokens

    #    Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges (in the same format that your BPE training code output)
    #    and (optionally) a list of special tokens. This method should accept the following additional parameters:
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    # Encode an input text into a sequence of token IDs.
    def encode(self, text: str) -> list[int]:
        # print("[bpe_tokenizer] PreTokenizing")
        pre_tokenizer = PreTokenizer()

        pretokens = pre_tokenizer.pretokens_from_text(text, self.special_tokens)
        # print("[bpe_tokenizer] PreTokenized")

        tokens: list[int] = []

        # print("[bpe_tokenizer] Encoding")
        for pretoken in pretokens:
            pretoken_tokens = self.__get_tokens_for_pretoken(pretoken)

            tokens.extend(pretoken_tokens)

        # print("[bpe_tokenizer] Encoded")

        return tokens

    def __get_tokens_for_pretoken(self, pretoken: tuple[bytes, ...]) -> list[int]:
        direct_hit = self.inverted_vocab.get(reduce(lambda x, y: x + y, pretoken))

        if direct_hit is not None:
            return [direct_hit]

        merged_set = self.__inner_get_tokens_for_pretoken(pretoken)

        return [self.inverted_vocab[record] for record in merged_set]

    def __inner_get_tokens_for_pretoken(self, pretoken: tuple[bytes, ...]):
        while len(pretoken) > 1:
            merge = None
            pretoken_offset = 0
            merges_offset = sys.maxsize

            for i in range(len(pretoken) - 1):
                candidate = (pretoken[i], pretoken[i + 1])

                hit = self.merges_dict.get(candidate)

                if hit is not None and hit < merges_offset:
                    merge = tuple([candidate[0] + candidate[1]])
                    pretoken_offset = i
                    merges_offset = hit

            if merge is None:
                break

            pretoken = pretoken[:pretoken_offset] + merge + pretoken[pretoken_offset + 2 :]

        return pretoken

    # Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
    # required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        remaining = ""

        while True:
            iterator = iter(iterable)
            try:
                chunk = next(iterator)
            except StopIteration:
                break

            chunk = remaining + chunk
            remaining = ""

            end_token_offset = -1
            for special_token in self.special_tokens or []:
                end_token_offset = chunk.rfind(special_token)

                if end_token_offset >= 0:
                    token_length = len(special_token)
                    yield from self.encode(chunk[: end_token_offset + token_length])
                    remaining = chunk[end_token_offset + token_length :]
                    break

            if end_token_offset == -1:
                remaining = chunk

        if remaining:
            yield from self.encode(remaining)

    # Decode a sequence of token IDs into text.
    def decode(self, ids: list[int]) -> str:
        result: bytes = b""

        for id in ids:
            value = self.vocab.get(id)

            if value is None:
                raise Exception("Invalid ID")

            result += value

        return bytes.decode(result, encoding="utf-8", errors="replace")
