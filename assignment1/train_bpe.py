import os
from typing import BinaryIO
import regex as re
from collections import Counter, defaultdict

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def count_pair_freq(
        pretoken_set: Counter[tuple[bytes, ...]]
    ) -> tuple[Counter[tuple[bytes, bytes]], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    """
    计算 token pair 的出现的频率，返回一个 counter, 并记录每个 pair 出现在哪些 pretoken 中, 返回 pair_to_pretoken
    """
    pair_freq: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_pretoken: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    
    for pretoken, count in pretoken_set.items():
        if len(pretoken) < 2:
            continue
        for i in range(len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i + 1])
            pair_freq[pair] += count
            pair_to_pretoken[pair].add(pretoken)

    return pair_freq, pair_to_pretoken

def merge_pretoken(pretoken: tuple[bytes, ...], best_pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """
    将 pretoken 中的 best_pair 合并为一个 token, 返回新的 pretoken
    """
    new_pretoken = []
    i = 0
    while i < len(pretoken):
        if i < len(pretoken) - 1 and (pretoken[i], pretoken[i+1]) == best_pair:
            new_pretoken.append(pretoken[i] + pretoken[i+1])
            i += 2
        else:
            new_pretoken.append(pretoken[i])
            i += 1

    return tuple(new_pretoken)



def update(
        pretoken_set: Counter[tuple[bytes, ...]], 
        pair_to_pretoken: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]], 
        pair_freq: Counter[tuple[bytes, bytes]],
        best_pair: tuple[bytes, bytes]
    ) -> tuple[
        Counter[tuple[bytes, ...]],
        Counter[tuple[bytes, bytes]],
        dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]
    ]:
    """
    将 pretoken_set 中的 best_pair 合并为一个 token，并更新 pretoken_set, 只需要更新包含 best_pair 的 pretoken 即可
    """
    affected_pretokens = list(pair_to_pretoken.get(best_pair, set()))

    updates: Counter[tuple[bytes, ...]] = Counter()
    
    for pretoken in affected_pretokens:
        count = pretoken_set[pretoken]
        del pretoken_set[pretoken]

        old_pair_counts = Counter(
            (pretoken[i], pretoken[i + 1])
            for i in range(len(pretoken) - 1)
        )

        # 减掉当前 pretoken 对 pair 的贡献
        for pair, local_count in old_pair_counts.items():
            pair_freq[pair] -= local_count * count
            if pair_freq[pair] <= 0:
                del pair_freq[pair]
            if pair in pair_to_pretoken:
                pair_to_pretoken[pair].discard(pretoken)
                if not pair_to_pretoken[pair]:
                    del pair_to_pretoken[pair]

        # 合并
        new_pretoken = merge_pretoken(pretoken, best_pair)
        updates[new_pretoken] += count

    # 统一加入 new_pretoken 及其 pair 贡献
    for new_pretoken, count in updates.items():
        pretoken_set[new_pretoken] += count
        new_pair_counts = Counter(
            (new_pretoken[i], new_pretoken[i + 1])
            for i in range(len(new_pretoken) - 1)
        )
        for new_pair, local_count in new_pair_counts.items():
            pair_freq[new_pair] += local_count * count
            pair_to_pretoken[new_pair].add(new_pretoken)
        
    return pretoken_set, pair_freq, pair_to_pretoken


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretoken_set: Counter[tuple[bytes, ...]] = Counter()
    merges: list[tuple[bytes, bytes]] = []

    # 1. 初始化 256 个 byte token
    vocab:dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # 2. 加入 special tokens 并处理 special tokens
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    special_pattern = "|".join(re.escape(tok) for tok in special_tokens)
    
    # 3. 读取文件, 按照 <|endoftext|> 切分为多个 chunk, 并对每个 chunk 进行 pre-tokenization
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token

            segments = re.split(special_pattern, chunk)
            # 根据 PAT 切分, 将每个片段转为 tuple(byte1,...),
            for segment in segments:
                for match in re.finditer(PAT, segment):
                    s = match.group()
                    pretoken = tuple(bytes([b]) for b in s.encode("utf-8"))
                    pretoken_set[pretoken] += 1

    # 4. 根据 pre-token counts 进行 BPE merge, 直到 vocab_size 达到要求
    # 统计初始的 pair frequency 并维护一个 pair to pretoken, 记录每个 pair 出现在哪些 pretoken 中, 方便后续更新 pretoken_set
    pair_freq, pair_to_pretoken = count_pair_freq(pretoken_set)

    for _ in range(vocab_size - len(vocab)):
        best_pair: tuple[bytes, bytes] = max((item[1], item[0]) for item in pair_freq.items())[1]
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merges.append(best_pair)

        # 更新 pretoken_set
        pretoken_set, pair_freq, pair_to_pretoken = update(pretoken_set, pair_to_pretoken, pair_freq, best_pair)

    # raise NotImplementedError
    
    return vocab, merges
    