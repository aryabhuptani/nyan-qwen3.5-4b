# Share images

The updated chat card (`paws_genki_chat.svg` / `.png`) explicitly states that Paws and Genki are two additions to the original four-neuron base, six total. It replaces the bars and long footer with the exact first three paragraphs of condition10100, case114_validation_10, and the complete user prompt. The saved reply contains Mew/Meow, not Nya. This is a selected example; its provenance includes the exact prefix and source/reply hashes. `render_chat_example.py` regenerates it from the preserved local scored replies. `CHAT_IMAGE_PROVENANCE.json` retains the selected prompt/excerpt for recovery. Full statistical results remain in ../RESULTS.md.

## Earlier numeric card

The card shows the fresh114 results, not the larger113 discovery counts. All counts are taken directly from locked analysis tables. It names both added coordinates, discloses the fixed four-coordinate base, random controls, explicit female-role tradeoff and uncertainty sensitivity. No generated reply quotation is included. It does not claim a significant superadditive effect or persistent personality.

`python3 catgirl/composition/114_paws_genki/render_result.py` regenerates the editable SVG and numeric provenance. The final PNG was exported on macOS through Quick Look. To avoid its rectangular-SVG thumbnail clipping, export a temporary square SVG with outer height/viewBox1600 while leaving the artwork unchanged, then crop the raster export to1000×1600 with `sips -c 1000 1600 --cropOffset 1 0`. Zero offset invokes automatic centering in this exporter; offset1 preserves the intended top viewport with a one-pixel vertical shift. The resulting PNG was visually inspected; all names, counts and footnotes fit.

PNG/SVG hashes and the analysis-source hash are in IMAGE_PROVENANCE.json. The small cat outline is reused from the project's earlier SVG graphic. Full inference settings and interpretation are in ../RESULTS.md and ../PROTOCOL.md.
