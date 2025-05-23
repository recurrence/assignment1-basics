from collections import Counter
import os
from cs336_basics.pre_tokenizer import PreTokenizer


class BPETokenizerTrainer:
    def __init__(self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        print(f"[bpe_trainer] PreTokenizing {self.input_path}")
        pre_tokenizer = PreTokenizer()

        frequencies = pre_tokenizer.frequencies_from_file(self.input_path, self.special_tokens)

        print("[bpe_trainer] PreTokenized")

        vocab = self.__get_initial_vocab()

        merges: list[tuple[bytes, bytes]] = []

        print("[bpe_trainer] Running Merges")
        while len(vocab) < self.vocab_size:
            merge_frequencies = self.__get_merge_frequencies(frequencies)

            new_vocab_pair = self.__get_top_pair(merge_frequencies)

            if new_vocab_pair is None:
                break

            merges.append(new_vocab_pair)

            new_vocab_entry = new_vocab_pair[0] + new_vocab_pair[1]

            vocab[len(vocab)] = new_vocab_entry

            # Update frequencies with the highest result in merge_frequencies
            for token, count in frequencies.copy().items():
                if self.__is_byte_pairs_in_target(new_vocab_pair, token):
                    frequencies.pop(token)

                    new_token: list[bytes] = []

                    i = 0
                    while i < len(token):
                        if i < len(token) - 1 and (token[i], token[i + 1]) == new_vocab_pair:
                            new_token.append(new_vocab_entry)
                            i += 2
                        else:
                            new_token.append(token[i])
                            i += 1

                    frequencies[tuple(new_token)] = count

        # print(vocab)

        return (vocab, merges)

    def __get_top_pair(self, merge_frequencies: Counter[tuple[bytes, bytes]]):
        top_pairs: list[tuple[bytes, bytes]] = []
        top_count = None

        for pair, count in merge_frequencies.most_common():
            if top_count is None:
                top_count = count

            elif top_count > count:
                break

            top_pairs.append(pair)

        if len(top_pairs) < 1:
            return None

        new_vocab_pair = max(top_pairs)

        return new_vocab_pair

    # Check if this pair exists anywhere in the target bytes
    def __is_byte_pairs_in_target(self, source_pair: tuple[bytes, bytes], target_bytes: tuple[bytes, ...]) -> bool:
        for i in range(len(target_bytes) - 1):
            target_pair = (target_bytes[i], target_bytes[i + 1])
            if source_pair == target_pair:
                return True

        return False

    def __get_merge_frequencies(self, frequencies: dict[tuple[bytes, ...], int]):
        merge_frequencies = Counter[tuple[bytes, bytes]]()

        for key, value in frequencies.items():
            for i in range(len(key) - 1):
                candidate = (key[i], key[i + 1])

                merge_frequencies[candidate] += value

        return merge_frequencies

    # Calculates the initial vocab of special tokens and utf8 bytes
    def __get_initial_vocab(self) -> dict[int, bytes]:
        encoded_special_tokens = list(map(lambda x: x.encode("utf-8"), self.special_tokens))

        initial_vocab = encoded_special_tokens + [bytes([i]) for i in range(256)]

        return {i: initial_vocab[i] for i in range(len(initial_vocab))}


if __name__ == "__main__":
    print("BPETokenizerTrainer: Starting")
    pre_tokenizer = BPETokenizerTrainer("data/simple.txt", 1000, ["<|endoftext|>"])
    results = pre_tokenizer.train()
    print(results[0])
    print(results[1])
    print("BPETokenizerTrainer: Finished")
