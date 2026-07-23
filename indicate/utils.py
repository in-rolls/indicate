from __future__ import annotations

import json
from collections.abc import Iterable

import torch

from .decoder import Decoder
from .encoder import Encoder

START_TOKEN = "^"  # noqa: S105 - seq2seq framing token, not a secret
END_TOKEN = "$"  # noqa: S105 - seq2seq framing token, not a secret

# Decoding stops at the END token; this is the safety cap if it never comes.
# Transliteration output length tracks input length, so the cap is input-relative
# (and still clamped by the model's absolute ``max_length_output``).
OUTPUT_LEN_MULTIPLIER = 2
OUTPUT_LEN_PADDING = 8


class CharTokenizer:
    """Character-level tokenizer holding the ``word_index``/``index_word`` maps.

    Loaded from the JSON files that were serialised by the original Keras
    ``Tokenizer`` so the vocabulary indices stay identical across the migration.
    """

    def __init__(self, word_index: dict[str, int]) -> None:
        self.word_index = word_index
        self.index_word = {index: word for word, index in word_index.items()}

    @property
    def vocab_size(self) -> int:
        # +1 for the reserved padding index (0).
        return len(self.word_index) + 1


def load_tokenizer(path: str) -> CharTokenizer:
    """Load a character tokenizer from a Keras-serialised tokenizer JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # The Keras export is a JSON-encoded string whose ``config.word_index`` is
    # itself a JSON-encoded string.
    if isinstance(data, str):
        data = json.loads(data)
    word_index = json.loads(data["config"]["word_index"])
    return CharTokenizer(word_index)


def sequence_to_chars(tokenizer: CharTokenizer, sequence: Iterable[int]) -> str:
    """Convert a sequence of indices back to characters, skipping padding (0)."""
    index_word = tokenizer.index_word
    return "".join(index_word[int(q)] for q in sequence if int(q) != 0)


def batch_candidates(
    words: list[str],
    input_lang_tokenizer: CharTokenizer,
    target_lang_tokenizer: CharTokenizer,
    encoder: Encoder,
    decoder: Decoder,
    max_length_input: int,
    max_length_output: int,
    beam_width: int = 1,
    mask_padding: bool = False,
) -> list[list[tuple[str, float]]]:
    """Ranked candidates for many words at once (the batched decode engine).

    Returns one ranked ``(text, score)`` list per input word, aligned to ``words``
    (empty/OOV words -> ``[]``). Inputs are padded to ``max_length_input`` exactly
    as the single-word path, so outputs are identical — just faster.
    """
    out: list[list[tuple[str, float]]] = [[] for _ in words]
    in_index = input_lang_tokenizer.word_index

    rows: list[int] = []
    id_lists: list[list[int]] = []
    caps: list[int] = []
    for i, word in enumerate(words):
        try:
            ids = [in_index[ch] for ch in word][:max_length_input]
        except KeyError:
            continue  # OOV word -> [] (matches the single-word path)
        if not ids:
            continue
        rows.append(i)
        id_lists.append(ids)
        caps.append(
            min(
                max_length_output, OUTPUT_LEN_MULTIPLIER * len(ids) + OUTPUT_LEN_PADDING
            )
        )
    if not rows:
        return out

    padded = [ids + [0] * (max_length_input - len(ids)) for ids in id_lists]
    inputs = torch.tensor(padded, dtype=torch.long)  # [M, max_length_input]
    src_mask = (inputs != 0) if mask_padding else None
    start_id = target_lang_tokenizer.word_index[START_TOKEN]
    end_id = target_lang_tokenizer.word_index[END_TOKEN]

    with torch.no_grad():
        enc_out, state = encoder(inputs)
        if beam_width and beam_width > 1:
            beam_out = _batch_beam(
                enc_out, state, start_id, end_id, decoder, caps, beam_width, src_mask
            )
            for row, hyps in zip(rows, beam_out, strict=True):
                out[row] = [
                    (sequence_to_chars(target_lang_tokenizer, t).strip(END_TOKEN), s)
                    for s, t in hyps
                ]
        else:
            greedy_out = _batch_greedy(
                enc_out, state, start_id, end_id, decoder, caps, src_mask
            )
            for row, toks in zip(rows, greedy_out, strict=True):
                out[row] = [
                    (
                        sequence_to_chars(target_lang_tokenizer, toks).strip(END_TOKEN),
                        0.0,
                    )
                ]
    return out


def _batch_greedy(
    enc_out: torch.Tensor,
    state: tuple[torch.Tensor, torch.Tensor],
    start_id: int,
    end_id: int,
    decoder: Decoder,
    caps: list[int],
    src_mask: torch.Tensor | None,
) -> list[list[int]]:
    n = enc_out.size(0)
    dec_input = torch.full((n, 1), start_id, dtype=torch.long)
    done = [False] * n
    outputs: list[list[int]] = [[] for _ in range(n)]
    for _ in range(max(caps)):
        logits, state = decoder(dec_input, enc_out, state, src_mask)
        pred = logits[:, -1].argmax(-1)  # [n]
        for i in range(n):
            if done[i]:
                continue
            pid = int(pred[i])
            outputs[i].append(pid)
            if pid == end_id or len(outputs[i]) >= caps[i]:
                done[i] = True
        if all(done):
            break
        dec_input = pred.unsqueeze(1)
    return outputs


def _batch_beam(
    enc_out: torch.Tensor,
    state: tuple[torch.Tensor, torch.Tensor],
    start_id: int,
    end_id: int,
    decoder: Decoder,
    caps: list[int],
    beam_width: int,
    src_mask: torch.Tensor | None,
) -> list[list[tuple[float, list[int]]]]:
    """Batched beam search: N words x k beams, per-word termination + caps.

    Length-normalized scores; each word's beams terminate and are capped
    independently, so results are identical to decoding the words one at a time.
    """
    n, s, u = enc_out.shape
    k = beam_width
    enc = enc_out.unsqueeze(1).expand(n, k, s, u).reshape(n * k, s, u).contiguous()
    mask = (
        src_mask.unsqueeze(1)
        .expand(n, k, src_mask.size(1))
        .reshape(n * k, -1)
        .contiguous()
        if src_mask is not None
        else None
    )
    h, c = state
    h = h.unsqueeze(2).expand(1, n, k, u).reshape(1, n * k, u).contiguous()
    c = c.unsqueeze(2).expand(1, n, k, u).reshape(1, n * k, u).contiguous()

    seqs = torch.full((n * k, 1), start_id, dtype=torch.long)
    scores = torch.full((n, k), float("-inf"))
    scores[:, 0] = 0.0
    finished: list[list[tuple[float, list[int]]]] = [[] for _ in range(n)]
    word_done = [False] * n
    base = torch.arange(n).unsqueeze(1) * k  # [n,1]

    for step in range(max(caps)):
        last = seqs[:, -1:]
        logits, (h, c) = decoder(last, enc, (h, c), mask)
        logp = torch.log_softmax(logits[:, -1, :], dim=-1).view(n, k, -1)
        vocab = logp.size(-1)
        cand = (scores.unsqueeze(-1) + logp).view(n, k * vocab)
        top_scores, top_idx = cand.topk(k, dim=-1)  # [n,k]
        beam_idx = torch.div(top_idx, vocab, rounding_mode="floor")
        tok_idx = top_idx % vocab
        global_beam = (base + beam_idx).view(-1)  # [n*k]
        seqs = torch.cat([seqs[global_beam], tok_idx.view(-1, 1)], dim=1)
        h = h[:, global_beam, :].contiguous()
        c = c[:, global_beam, :].contiguous()
        scores = top_scores.clone()
        length = seqs.size(1)

        for i in range(n):
            if word_done[i]:
                continue
            for b in range(k):
                if int(tok_idx[i, b]) == end_id:
                    gi = i * k + b
                    finished[i].append(
                        (float(scores[i, b]) / length, seqs[gi, 1:].tolist())
                    )
                    scores[i, b] = float("-inf")
            at_cap = (step + 1) >= caps[i]
            if bool(torch.isinf(scores[i]).all()) or at_cap:
                if not finished[i]:  # cap hit with nothing finished -> best live beam
                    b = int(scores[i].argmax())
                    finished[i].append(
                        (float(scores[i, b]), seqs[i * k + b, 1:].tolist())
                    )
                word_done[i] = True
        if all(word_done):
            break

    for i in range(n):
        if not finished[i]:
            b = int(scores[i].argmax())
            finished[i].append((float(scores[i].max()), seqs[i * k + b, 1:].tolist()))
        finished[i].sort(key=lambda x: x[0], reverse=True)
    return finished
