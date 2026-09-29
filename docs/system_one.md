# System One, Jev, and this repo

[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is TypeSafe's first System One model (September 2026). The class is named for Kahneman's fast System 1 thinking: a decision, not a deliberated essay.

## What the hosted model does

You POST a `state` (text, or structured program state) and a map of questions. Each question is one of:

| Type | Asks | Returns |
|------|------|---------|
| Choice | which named option? | `choice`, `probabilities`, `confidence` |
| Score | where on an ordered rubric? | `score` (probability-weighted level), `probabilities`, `confidence` |
| Noul | is this true? | `noul` = P(yes) |

The answers come from one query. There is no token loop, so there is no string to parse and no way for the model to emit a type that was not in the question. TypeSafe trains with a method they call RLCD (reinforcement learning for calibrated decisions). The architecture and the RLCD algorithm are not public.

## What "openjev" is

Independent servers implement the same HTTP shape and read probabilities out of an existing open model (DiffusionGemma, Qwen, and others). They are useful if you want the API on hardware that can hold those weights. They are not a from-scratch model, and a 26B-class checkpoint does not fit an 8 GB card.

## What you train here

A bidirectional Transformer encoder plus three heads, fixed at training time:

- topic → choice over `chat` / `ops` / `learning`
- urgency → score over `calm` / `soon` / `now`
- needs a reply → noul

Confidence in this course is `1 - H(p) / log(k)`. That matches the public description (confidence comes from the shape of the distribution). It is not a claim about TypeSafe's unpublished formula.

Jev can be asked a question it was not given a separate head for. This model cannot. With a few million labeled bytes, the honest trainable object is a small set of heads you defined. Arbitrary questions at request time need a pretrained language model under them, which is the OpenJev approach, not this course.

## Where a Discord log fits

A chat line is a state. You decide what is worth branching on in code: topic, urgency, whether something expects an answer. The model returns numbers your program can threshold. It does not write the message back. Writing the message is buka (from scratch, small LM) or buka-evo (LoRA on Ministral).
