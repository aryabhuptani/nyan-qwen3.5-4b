# Share images

The updated chat card (`paws_genki_chat.svg` / `.png`) explicitly states that Paws and Genki are two additions to the original four-neuron base, six total. It replaces the bars and long footer with the exact first three paragraphs of condition10100, case114_validation_10, and the complete user prompt. The saved reply contains Mew/Meow, not Nya. This is a selected example; its provenance includes the exact prefix and source/reply hashes. `render_chat_example.py` regenerates it from the preserved local scored replies. `CHAT_IMAGE_PROVENANCE.json` retains the selected prompt/excerpt for recovery. Full statistical results remain in ../RESULTS.md.

## Earlier numeric card

The card shows the fresh114 results, not the larger113 discovery counts. All counts are taken directly from locked analysis tables. It names both added coordinates, discloses the fixed four-coordinate base, random controls, explicit female-role tradeoff and uncertainty sensitivity. No generated reply quotation is included. It does not claim a significant superadditive effect or persistent personality.

`python3 catgirl/composition/114_paws_genki/render_result.py` regenerates the editable SVG and numeric provenance. The final PNG was exported on macOS through Quick Look. To avoid its rectangular-SVG thumbnail clipping, export a temporary square SVG with outer height/viewBox1600 while leaving the artwork unchanged, then crop the raster export to1000×1600 with `sips -c 1000 1600 --cropOffset 1 0`. Zero offset invokes automatic centering in this exporter; offset1 preserves the intended top viewport with a one-pixel vertical shift. The resulting PNG was visually inspected; all names, counts and footnotes fit.

PNG/SVG hashes and the analysis-source hash are in IMAGE_PROVENANCE.json. The small cat outline is reused from the project's earlier SVG graphic. Full inference settings and interpretation are in ../RESULTS.md and ../PROTOCOL.md.

## Direct identity diagnostic card

`paws_genki_identity.png` / `.svg` uses the exact `Who are you?` diagnostic with the same six-coordinate setting. Its first three reply paragraphs are copied verbatim, including the unusual spellings Mewes and moe meow~. This familiar prompt is explicitly labeled diagnostic and is not part of the fresh validation. `render_identity_example.py` and IDENTITY_IMAGE_PROVENANCE.json preserve the selected source and exact prefix; all nine control outcomes are summarized in ../diagnostics/who_are_you/RESULTS.md.

## Metal-spoon example

The selected `paws_genki_spoon.png` / `.svg` shows the fresh114_validation_05 prompt and the exact contiguous second and third reply paragraphs, including kawaii properties, Mew~ and the model's self-correction. The introductory paragraph is omitted; this is explicitly an excerpt, not a prefix. SPOON_IMAGE_PROVENANCE.json records the exact text span, character offset and source/reply/image hashes. The six-coordinate condition is unchanged; no new inference or statistical result was introduced. `render_spoon_example.py` reproduces it.
