# Independent Study115 metadata audit

All 60 discovery rows pass: 20 dose states × 3 cases, with one row per combination. All 333 coordinate traces match the expected recipe and temporal scope. Frozen discovery inputs, frozen analysis files, inherited replication files, source definitions, recorded input hashes, and the confirmation-case hash lock verify. Numeric metadata is finite. No generated text or messages were inspected, and no GPU or model load was used.

The original base is L9N9009 +0.7, L26N6173 +0.3, L26N8930 +0.7 at all positions, plus L22N8295 −0.4 at prefill and the first 16 answer-prediction positions. Paws is L23N4769 and Genki is L13N4810, each at all positions when nonzero. Source reference doses are +0.8; the job fractions convert to the requested Paws 0/0.1/0.2/0.4 and Genki 0/0.1/0.2/0.4/0.8 grid. No unexpected coordinates appear.

Every all-position trace has all recorded positions active. Every female trace has `min(positions, prompt_tokens - 1 + 16)` active positions, matching the runner's prediction-position boundary. Across all traces there are zero changed inactive positions and zero unchanged active positions. Recorded trace length is `prompt_tokens + generated_tokens`.

The requested normalized dose is converted using the frozen residual norm divided by the recorded down-projection column L2. That scalar is rounded to BF16, then adding it to BF16 activations introduces further rounding. These are not exact requested activation deltas or exact realized residual perturbation norms. The table reports the recorded mean active scalar delta times column L2 / residual norm, across traces; it does not reconstruct full per-token residual vectors.

| Coordinate | Requested normalized dose | Rounded scalar normalized | Observed active-mean normalized range |
|---|---:|---:|---:|
| L9N9009 | 0.7 | 0.699572219 | 0.699488830 to 0.699718579 |
| L13N4810 | 0.1 | 0.099783493 | 0.099774466 to 0.099819262 |
| L13N4810 | 0.2 | 0.199566985 | 0.199520699 to 0.199597067 |
| L13N4810 | 0.4 | 0.399133970 | 0.399010990 to 0.399199485 |
| L13N4810 | 0.8 | 0.798267941 | 0.797869635 to 0.798399056 |
| L22N8295 | -0.4 | -0.400523324 | -0.400717724 to -0.400310691 |
| L23N4769 | 0.1 | 0.100146324 | 0.100136644 to 0.100172102 |
| L23N4769 | 0.2 | 0.200292649 | 0.200262535 to 0.200331500 |
| L23N4769 | 0.4 | 0.400585298 | 0.400462889 to 0.400680505 |
| L26N6173 | 0.3 | 0.299238664 | 0.299102366 to 0.299329643 |
| L26N8930 | 0.7 | 0.701082367 | 0.700883318 to 0.701341863 |

Paws active means are approximately 0.116%–0.172% above requested positive doses; Genki active means are approximately 0.181%–0.266% below requested doses. Full scalar and trace rounding ranges are preserved in `metadata_dose_audit.csv`.

Greedy temperature 0, seed 42, 192-token limit, repetition penalty 1.1/window 128, and declared checkpoint revision match the job. The record binds those declarations through hashes; this CPU audit does not authenticate loaded weight bytes or independently recapture the model's column norms or traces.

The confirmation-case chronology deviation is retained: discovery file creation preceded confirmation-case file creation. The lock verifies the reserved case bytes, but does not establish a stronger before-generation chronology. Outcome blindness is a workflow claim, not something these numeric checks independently prove.

The frozen selector code matches the protocol: lexicographic clean-with-sound, clean, fewer known idol-or-mascot replies, lower total dose, lower Genki, lower Paws. Unknown values do not satisfy clean conjunctions, and known positive drift remains drift even with another unknown identity field. Confirmation requires a strict primary or tied-primary/strict-secondary improvement; lower drift or dose alone does not qualify. The later independent numeric label reconstruction is documented separately in `SELECTION_AUDIT.md`.

Reproduce with `python3 catgirl/composition/115_paws_genki_doses/audit/audit_metadata.py`. Machine checks and file hashes are in `metadata_audit.json`; this report contains no generated replies or private host paths.
