# Persistent-Effort Adaptation v0.1

This is the **poor-man compute layer** for MYCELIUM.

The system assumes compute is scarce but persistence is cheap. It therefore refuses to retrain a giant generator for every preference or song. It keeps the acoustic model mostly frozen and learns around it.

## Canonical rule

> Do not spend gradient compute to relearn what durable memory can remember.

The adaptation ladder is:

```text
OWNED / LICENSED SONG INPUT
        |
        v
SHA-256 + RIGHTS + AUDIO PROBE
        |
        v
PERSISTENT SONG MEMORY
        |
        v
REFERENCE-CONDITIONED GENERATION
        |
        v
SHORT SEEDED PREVIEWS
        |
        v
UCB PARAMETER SEARCH
        |
        +----> HUMAN / AI-EAR SCORE ----+
        |                               |
        +<-------- MEMORY UPDATE <------+ 
        |
        v
PROMOTE ONLY WINNERS TO FULL SONG
        |
        v
REPAINT ONLY FAILED REGIONS
        |
        v
PLATEAU TEST
        |
        +-- still improving --> keep frozen model
        |
        +-- plateaued --------> gradient sensitivity estimate
                                  |
                                  v
                          tiny targeted adapter
```

## Why this is different from ordinary training

Traditional personalization spends repeated forward + backward passes updating many parameters. Persistent-Effort Adaptation treats **inference, memory, search, and localized repair** as the default learning mechanism.

The expensive training step is gated behind evidence that the cheaper loop has plateaued.

That gives us five compute advantages:

1. **One input song becomes reusable memory.** Its hash, rights state, path, generation history, winning seeds, and control settings survive model restarts and upgrades.
2. **Preview before render.** Explore with short clips; spend full-song inference only on winners.
3. **Never repeat failed search blindly.** UCB1 remembers which generation-control combinations have actually worked, and in-flight/unscored experiments reserve their arms so a synchronous batch explores distinct untried configurations before repeating one.
4. **Repair instead of restart.** ACE-Step repaint regenerates only a failed time interval.
5. **Sparse training only after plateau.** ACE-Step Side-Step gradient sensitivity can identify the projections that respond to the dataset; a small rank adapter is then tested against the frozen baseline.

This does **not** claim that a new training algorithm has already beaten full fine-tuning in controlled experiments. It is a compute-allocation architecture. Its superiority must be demonstrated by the benchmark protocol below.

## Requirements

- Python 3.11+ recommended
- ACE-Step 1.5 API server running locally, default `http://127.0.0.1:8001`
- `ffprobe` is optional but recommended for metadata probing
- Song input must be owned, licensed, or public domain

The control layer itself uses only the Python standard library.

## Quick start

From the repository root:

```bash
python -m persistent_effort.engine health
```

Ingest a song once:

```bash
python -m persistent_effort.engine ingest "D:\\Music\\reference.wav" --rights owned
```

The command returns a stable `song_id` derived from the source SHA-256.

Generate eight cheap 30-second probes:

```bash
python -m persistent_effort.engine batch song_0123456789abcdef \
  --prompt "dark cinematic Midwest hip-hop, human vocal, raw dynamics" \
  --lyrics "[Verse]\n..." \
  --count 8 \
  --duration 30
```

Score each result from `0.0` to `1.0`:

```bash
python -m persistent_effort.engine score exp_abc123 0.91
python -m persistent_effort.engine score exp_def456 0.42
```

Or record direct preference. Pairwise confidence is converted into complementary winner/loser rewards so this path updates both UCB learning and adaptation evidence:

```bash
python -m persistent_effort.engine pair exp_abc123 exp_def456
```

See remembered winners:

```bash
python -m persistent_effort.engine top song_0123456789abcdef
```

Promote a winner to a full render:

```bash
python -m persistent_effort.engine promote exp_abc123 --duration 210
```

Repair only seconds 61-76:

```bash
python -m persistent_effort.engine repaint exp_full123 \
  --start 61 --end 76 \
  --instruction "keep arrangement, improve vocal timing and intelligibility"
```

Ask whether training is justified:

```bash
python -m persistent_effort.engine adaptation song_0123456789abcdef
```

Until there is enough evidence of plateau, the answer is deliberately `keep_searching` or `freeze_weights`.

## Frontier sparse-adaptation rung

ACE-Step 1.5 Side-Step now supports gradient sensitivity estimation. The intended escalation is:

1. Build a tiny dataset from **winners plus representative failures**, not the entire catalog.
2. Preprocess it once.
3. Run sensitivity estimation over a few batches.
4. Target only high-response projections/modules.
5. Start with rank-16 LoRA for current inference compatibility.
6. Test LoKR as an experimental speed path only when its custom inference path is wired and verified.
7. Keep the adapter only if it beats the persistent frozen baseline in blind A/B tests.

The reason for starting with rank-16 LoRA is operational: current ACE-Step tooling has the safest load path for PEFT LoRA. LoKR can train much faster, but its inference integration must be explicitly verified rather than assumed.

## Benchmark protocol

The method earns the word **better** only if it wins measured tests.

For the same 10 reference songs and the same prompts:

| Metric | Frozen + random search | Full ordinary LoRA | Persistent-Effort |
|---|---:|---:|---:|
| GPU-minutes consumed | measure | measure | measure |
| Peak VRAM | measure | measure | measure |
| Human blind preference | measure | measure | measure |
| AI-Ear technical score | measure | measure | measure |
| Reference-style adherence | measure | measure | measure |
| Novelty / non-copy distance | measure | measure | measure |
| Number of complete-song renders | measure | measure | measure |
| Repaint seconds vs regenerated seconds | measure | measure | measure |

Primary success criterion:

```text
quality_per_gpu_minute(Persistent-Effort) > quality_per_gpu_minute(best baseline)
```

Secondary criterion: quality must not collapse when the ACE-Step base checkpoint is swapped, because the durable memory is stored outside the weights.

## State model

SQLite runs in WAL mode. The important tables are:

- `songs` — immutable content identity and rights state, plus the latest verified local source path and audio metadata
- `arms` — generation-control configurations and accumulated rewards
- `experiments` — every seed, request, result, hash, status, and score
- `pairwise_feedback` — winner/loser judgments
- `adaptation_events` — evidence and recommendations when sparse training is considered

Default state path:

```text
~/.mycelium/persistent_effort.db
```

Default artifacts:

```text
~/.mycelium/persistent_effort/artifacts/<song_id>/<experiment_id>/mix.wav
```

## Failure behavior

- Missing source song: fail closed.
- Unauthorized rights state: reject ingest.
- Re-ingesting identical bytes from a valid new location refreshes the durable path; conflicting rights-state relabeling is rejected.
- ACE-Step unavailable: no experiment is falsely marked complete.
- Backend failure: persisted as `failed` with error text.
- Download failure: no artifact hash is committed.
- Cross-origin absolute artifact URLs never receive the ACE-Step bearer token.
- Duplicate song bytes: reuse the same canonical song identity.
- Feedback for queued, running, failed, or artifactless experiments is rejected.
- Scalar and pairwise feedback claim every unscored result inside the same database transaction as the arm update; concurrent losing writers abort without changing scores, pairwise history, or learning totals.
- Repaint without completed source: rejected.
- Training recommendation without enough evidence: rejected by policy through `keep_searching`.

## What comes next

The next integration target is `MASSIVEMAGNETICS/the-ai-ear`: use its music/emotion analysis as an **auxiliary score**, while human preference remains the final reward. After that, add cached structural fingerprints and section-level retrieval so memory can transfer winning strategies across related songs without weight updates.
