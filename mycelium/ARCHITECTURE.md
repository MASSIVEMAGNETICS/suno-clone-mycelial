# MYCELIUM Architecture v1.0

## Core Innovation

MYCELIUM introduces a **VSA Semantic Layer** on top of modern neural audio models.

Traditional systems (Suno, Bark, MusicGen) rely almost entirely on statistical pattern matching over audio tokens. They are excellent at local acoustic quality but weak at clean, long-range symbolic structure and true compositionality.

MYCELIUM fixes this by inserting a Vector Symbolic Architecture (VSA) layer that handles:
- Musical roles and structure (verse, chorus, motif development)
- Emotional arcs and tension/release
- Binding of lyrics to musical ideas
- Long-range coherence through explicit symbolic operations

## Layered Architecture

1. **VSA Semantic Layer** (this package)
   - Hyperdimensional vectors + circular convolution binding
   - SongPlan creation from prompt + lyrics
   - High-level musical discourse planning

2. **Conditioning Bridge**
   - `vsa_plan_to_conditioning()` converts symbolic plans into parameters
   - Feeds Bark, MusicGen, XTTS, RVC, and procedural engine

3. **Acoustic Rendering Layer**
   - Bark / XTTS for vocal generation (with RVC voice locking)
   - MusicGen or procedural engine for adaptive accompaniment
   - Full stem separation and export

4. **Interface Layer**
   - Single-file `index.html` creative surface
   - Mycelium graph visualization
   - Live stem mixing and quantum parameters

## Why Circular Convolution?

Circular convolution as a binding operation offers better approximate invertibility and similarity preservation than simple element-wise multiplication. This is particularly valuable in music, where we want motifs and emotional states to remain recognizable after being bound into larger structures.

## Future Directions

- Deeper cross-scale binding between VSA semantic vectors and acoustic token layers
- Magnetics-inspired dynamics (attraction/repulsion between musical ideas)
- Self-improving ITEM_MEMORY from user generations
- Full fractal multi-resolution token streams

This architecture is designed to grow with the artist while staying completely sovereign.