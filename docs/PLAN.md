# Plan

Teach a from-scratch System One model in the same notebook style as buka and buka-evo, sized for a GTX 1080 and a small labeled chat set.

| Phase | Deliverable |
|-------|-------------|
| Decisions | Notebooks 02 and 07: choice, score, noul, confidence |
| Encoder | Notebooks 04–06: bidirectional attention, block, pool |
| Train | Notebooks 08–10 and `scripts.train` |
| Serve | Notebook 11 and the local page |

## Out of scope

- Reproducing Jev's hosted weights or RLCD
- Serving a 26B model behind a Jev-compatible HTTP API (that is what OpenJev projects do)
- Generating chat text (buka, buka-evo)
