# What you can train

## The machine

GTX 1080, 8 GB, compute capability 6.1. Train in float32. Pascal has no tensor cores; float16 saves a little memory and does not speed the matmul up. CUDA 11.8 wheels match the other two repos. CUDA 13 drops this card.

AdamW in float32 is about 16 bytes per parameter (weight, gradient, first moment, second moment), plus activations. A 14M model is a few hundred megabytes. The 8 GB card still has room at that size.

## The data

A local monologue export of the Luka chats is on the order of **3 MB** of text, a few million bytes. Chat-pair JSONL next to it is smaller. That is the corpus. Chinchilla-style language-model sizing would say "a tiny net, many epochs." This course is not a language model. It is a classifier with a Transformer body, so the relevant count is **labeled states**, not raw bytes.

The committed sample is a few dozen unique labeled lines. `needs_reply` is not "does the line contain a question mark": some questions are status updates, and some statements still need an answer. Topic and urgency are not a single keyword either. Tiny can still memorize a file this small. Your own export becomes useful when you label states in the same schema (`state`, `topic`, `urgency`, `needs_reply`). A few thousand careful labels is a real training set for `tiny`. Tens of thousands is when `small` starts to earn its width. `stretch` is there so you can see a wider encoder still fit; it will not invent Jev-level judgment from a chat log.

## Presets

| Preset | Depth × width | Params | Role |
|--------|----------------|--------|------|
| tiny | 4 × 128 | 0.90M | default |
| small | 6 × 256 | 5.2M | more labels |
| stretch | 8 × 384 | 14.3M | still easy on the 1080 |

Byte vocabulary is 256 plus one pad id. Context is 128 tokens on tiny (a short message, not a long thread). Pack a thread by keeping the last bytes if you need more history than that.

## What not to train from scratch on this box

- A 120M chat LM on 3 MB of Discord. That is a VRAM demo. buka already has the architecture; the text is too small for those weights to learn English.
- A 3B model from scratch. buka-evo and the local assistant in `../eve` start from Ministral and adapt it.
- Jev itself. The weights are hosted, and the training recipe is not public.
