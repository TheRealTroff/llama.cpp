# Vision gate (exp/vision-gate) - state as of 2026-09-27

Projector: `/Users/troff/play/qwen3.8-mmproj-F16.gguf` (Unsloth, qwen3vl_merger, 461M, out 5120). Loads with both
pick files. Inputs: `images/preprocess.sh` (HEIC/WebP via sips, EXIF rotation baked, sRGB, alpha dropped, ladder
full/2048/1024/512). One image token per 32x32 px; the loader caps 8..4096 tokens per image.

## Reference set (CLI, plain build, no drafter, greedy, thinking off) - refs/sep27-cli/{q4,ud}
Runner `run-vision-ref.sh`, scorer `score.py`, prompts `prompts.tsv`. 49 rows per line (afternoon: + label full rung, + IMG_2924 chalkboard menu, beach checklist set). Verdicts identical on
both lines. SIGNED OFF by the owner 2026-09-27 ("I can't find anything off about your facts"): refs/sep27-cli is the vision reference for both lines.

PASS both lines: table card 702 at every rung incl. 192 tokens; magazine headline name; diagram layer counts
48/27 (full, 1024) and psi = SiLU (full, 2048, 1024); weather low-on-14th = 13 (full, 1024), storms 29/30;
every checklist row (meal 3/3, surf camp 2/2, packing 3/3, diagram comparison 4/4); CDG sign paragraph
1.000 at all four rungs AND the unrotated control; the SYNTHETIC sign (text that exists nowhere) 1.000 at all
four rungs on both lines, byte-identical across rungs -> the OCR is real at 192 tokens, not recall.
Bottle label: FAIL at 1024 ("St. Germain"), PASS at full ("St. Feuillien") on both lines = the one row that needs 4096 tokens.
Handwritten chalk menu (IMG_2924): Garden Tonic $15 and wine 14/42 PASS at 768 tokens both lines; at 192 tokens both
confabulate a neighbour ("Flying Squirrel/Sausage, $12" from "Flying Saucer"); the sober-order checklist 4/4 on ud, 3/4 on q4
(q4 skips the mocktails, misreads Natalie's as Namalie's, invents a $1 outside-food fee) = chalk at 768 tokens is where the
lines start to differ in reading, not only in wording.
FAIL both lines:
psi at 512 ("Gating"/"Gated Delta Rule"); low-on-14th at 512 (14); beach event = "solar eclipse viewing"
(q4 invents the April 2024 eclipse and California). Owner: the Quiksilver Festival surf competition, La Nord, Hossegor,
2026-09-20; checklist = surf + competition|contest|festival|championship (nothing in frame names the brand); 0/2 both lines.
Cross-line: 15/37 answers byte-identical (the short ones); long answers differ = two quantizations, not a signal.
ud at full/2048/unrot transcribes WITH the sign's line breaks and hyphens ("exactly as written"); scorer undoes
printed hyphenation before comparing.

## Encoder cost (F16 projector, Metal, this box) - superlinear in tokens
| rung | tokens | encoder | est. 27B prefill @137 t/s | served stall (all slots) |
| 512  |  192 | 0.22 s |  1.4 s |  ~2 s |
| 1024 |  768 | 0.89 s |  5.6 s |  ~7 s |
| 2048 | 3072 | 7.1 s  | 22 s   | ~30 s |
| full | 4096 | 11.5 s | 30 s   | ~40 s |
Nothing measured gains from 2048/full except (unverified) the bottle label -> propose `--image-max-tokens 1024`
for the served pick.

## Server facts (traced in prod tools/server/server-context.cpp, 2026-09-27)
--mmproj loads once; same slots, same chat endpoint; image_url = base64 or http URL (server downloads).
Loading it flips every slot to has_mtmd: ctx_shift and n_cache_reuse disabled server-wide (warned), no
checkpoint after an image chunk. The image chunk is embeddings, so it cannot ride the shared token-id batch:
the slot encodes and issues a PRIVATE llama_decode (mtmd_helper_decode_image_chunk), every other slot waits for
encoder + image prefill. The spec state is passed into that call via callback = the DFlash drafter sees image
batches, UNGATED. Image tokens carry 2D mrope positions -> benign find_slot non-consecutive warnings; ctx
classes / checkpoints / prompt cache never exercised with them.

## Traps found today
- llama-mtmd-cli is not rebuilt by the mint; a stale binary aborts in arg parsing (struct layout). Rebuild target.
- The CLI has no thinking switch and --chat-template-file is not registered for it: pass the template TEXT
  with --chat-template (chat-template-nothink.jinja = embedded template, enable_thinking forced false).
- The CLI has no single-turn text mode: -p without --image drops into chat mode and loops on stdin (373 MB
  of prompts in 5 min). The text-only control is server-phase only.
- Never launch a second pass with a TAG whose summary exists (fixed: the runner appends now); kill runners by
  PID - killing only the CLI child makes the loop move to the next row.

## Open (server phase)
1. Serve the pick with --mmproj (plain, spec off) and compare shas with refs/sep27-cli; then spec on.
2. Text-only control on a projector-loaded server.
3. Multi-slot: one image slot + text slots (stall vs tokens); several image slots (scratch/route bugs).
4. --image-max-tokens 1024 as the served default; bottle label at full.
5. Optional: thinking arm on the abstract rows (-n 1500) for drafter acceptance on long image-conditioned text;
   encoder-pooled features as an image-similarity probe (small tool against libmtmd, no hook today).
