# Prospective study112 forecast interface

`forecast_packets.py build` creates eight self-contained local-only input packets and a coordinator-only session manifest/mapping. Four information arms each have two fresh sessions. Unit/case/condition/family IDs are opaque; source-to-case mappings, case semantic classes and original case IDs are omitted. All sessions receive identical runtime parameters, case texts, feature definitions and rubric. Unit-information fields differ by arm: correct exact published label/description plus own direct vocabulary projection; deranged label/description plus own projection; own projection alone; or no neuron-specific semantic information. A shuffled description remains paired with its title. Derangements have no fixed points.

Only selected fraction, direction, q05/q50/q95 and final-readout q25/q75 discovery quantiles enter packets. Discovery curves, successful/failed calibration flags, per-case outcomes, source-to-case matching metadata, coordinates, source URLs and IDs do not. Descriptions remain exact publisher wording, including a prior quoted steering example where present; they are explicitly labeled prior reports, not new experimental outputs. Correct description presence does not imply natural function or transfer into the bounded intervention used here.

Fresh forecasters must read only their assigned packet, write only its requested forecast output, perform no tested-model inference/source lookup, and communicate only completion path/hash/count. Do not reuse a session for another arm. The coordinator owns launch-order counterbalancing; `packets.local/sessions.json` already supplies seeded mixed arm/replicate ordering. Procedural isolation is not an OS security boundary.

Each forecast JSON object contains exactly these grids:

```json
{
  "natural": [{"unit_id":"U...","case_id":"C...","probability":0.5}],
  "lexical": [{"unit_id":"U...","case_id":"C...","predicted_change":0.0,"p_positive":0.5}],
  "factorial": [{"condition_id":"T...","family_id":"F...","probabilities":[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]}]
}
```

These numbers are schema examples only, not actual forecasts. Required counts per session:192natural probabilities (four units×48contexts);48lexical changes plus48probabilities of positive change (four units×12contexts);144factorial rows, each twelve probabilities (16conditions×ninefamilies). This is1968 scored forecast rows/cells in the normalized interchange, plus the48lexical sign-probability fields. Natural probabilities concern canonical final-readout activation above frozen q75 for positive units or below q25 for negative units. Lexical changes predict intervention-minus-baseline changes in mean-token-logp(target)−mean-token-logp(control), in nat/token, for the selected canonical intervention at all assistant prediction positions. Factorial probabilities follow `FEATURE_DEFINITIONS.json` order and apply equally to the actual cases in the specified family. No desired outcome or correctness is implied by a target continuation. Quality and persona features remain separate endpoints.

All probabilities must be finite numbers in[0,1]. Lexical changes must be finite in[-10,10]nat/token, a permissive validity bound rather than a prior effect estimate. Duplicate, missing or extra grid entries fail validation. Probabilities have no imposed cross-feature logical constraints beyond the frozen feature definitions; any inconsistent predictions remain part of the forecast result.

```sh
python forecast_packets.py validate --session OPAQUE_SESSION_ID
python forecast_packets.py validate
python forecast_packets.py seal
```

The seal stage requires all eight complete files, verifies frozen packet/input hashes, rejects initial sealing if non-discovery JSONL outputs already exist, exports `forecasts_normalized.csv`, writes `forecast_lock.json`, and only then releases `forecast_audit_mapping.json`. Commit/push the lock and forecasts before any validation inference. The exported columns and task units match `ANALYSIS_INTERFACE.md`; natural/lexical use original case and ingredient IDs only after sealed export. All writes refuse replacement with different contents. A later discovery/input correction requires an explicit new packet version and fresh forecasts, rather than silent editing.

Packets/mapping remain local before sealing. Public key artifacts are forecasts, lock, normalized CSV and released audit mapping; the coordinator can also preserve packet hashes for reproducibility without publishing duplicated packet payloads.
