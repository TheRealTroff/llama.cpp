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

## Server phase (2026-09-27 afternoon) - run-vision-server.sh, results vision-srv-sep27-{q4,ud}-{base,spec}
The served pick (manifest env, Turbo4 KV, ctx 102400) with --mmproj and the same no-think template text.
- base arms (no drafter): every row completes; the text-only control = "Paris"; 768-token image = 6.4-7.1 s prompt,
  4096 tokens = 34-42 s. Verdicts vs the checklists match the CLI on the short facts; the long answers differ in
  wording from the CLI refs (20/49 q4 and 14/49 ud shas equal) = the pick's numerics class (SoA/acch prefill,
  Turbo4 KV) vs the plain CLI kernels, the same cross-class gap as the text lineage. THE SHA GATE FOR THE SERVED
  PICK IS base-arm vs spec-arm (same class); the CLI refs are the correctness (checklist) reference.
- spec arms (the full pick) on the prod build: TWO DEFECTS. (1) after any image the drafter never drafted:
  the server handed it slot.prompt.n_tokens() (804) as its position while its cache held the mrope grid
  positions (67) -> every draft decode rejected, 15 t/s undrafted, silent. (2) images above n_batch
  (3072/4096 tokens) overflowed the windowed draft KV during ingestion and the whole request died HTTP 500
  (14 rows per line). FIXED on the branch (3a4aae0f1): drafter position = pos_next(); a failed ingestion
  drops speculation for that request (spec_off_request) with one warning. After the fix (q4): acceptance
  44-64% on 300-token answers after a 768-token image (meal 300 tok 20 s -> 11.5 s, sober 9.8 s); short facts
  byte-identical to base; one long fork = a controller pick (gone at PICK_SPEC_EV=0), one = a 0.104-nat tie
  (' T'onic vs ' or', base-arm top-2 logprobs) = the rounding band.
  FULL FIXED SPEC ARMS (vision-srv-fix-{q4,ud}-spec, summaries in refs/sep27-srv): shas vs the base arm 42/50 (q4) and
  46/50 (ud) identical; every short fact identical; every checklist verdict identical (35/5, 34/6); the differing
  rows are long answers under the depth controller. Generation with an image in context (t/s, base -> spec):
  q4 512-rung 15.1 -> 33.7 (acc 69%), 1024-rung 15.0 -> 29.1 (acc 65%); ud 13.9 -> 32.7, 13.7 -> 25.8 (acc 62-70%);
  rows above n_batch (3072/4096 tokens) generate undrafted (14 per line, warned once each) until the draft KV is
  windowed during ingestion; a 1770-token full-rung row drafts at 71%. The large-image cause FOUND: the dflash drafter's memory is an iswa cache whose SWA part holds
  n_swa + n_ubatch = 2048 + 512 = 2560 cells (5 layers, 50 MiB); position-based windowing cannot bound image
  cells (an image's cells share a few mrope positions), so a 3072-token chunk never fit. FIX (branch, pending
  gate): with --mmproj the server grows the drafter's n_ubatch (and n_batch) to the image cap (4096 default,
  --image-max-tokens if set) -> SWA cache 6144 cells, ~+70 MiB; one injection slice per image.

## Marginal-band reading set (prompts-menu.tsv, refs/sep27-menu) - the first task-level quality number between the lines
Every chalk-menu item at 768 tokens (hard) and 3072 tokens (control), CLI plain build. Misread rows (a wrong
digit, a dropped word or a misspelled name counts): 768 tokens q4 9/16 vs ud 7/16 (q4's extra two = a chip
price and the stout's ABV, both small digits; on the beer row q4 also misprices where ud is right);
3072 tokens 3/16 on both, IDENTICAL answers (Rose omitted, San Benedetto missed, "Contreau"). Same direction
as the sober-order row and the KLD ordering (UD closer to bf16), but 16 items and a two-row gap = suggestive.
More handwritten boards would make it a trend. The owner's point: KLD says the lines differ, this says what
that costs in what a person sees.

## Open (server phase)
1. Serve the pick with --mmproj (plain, spec off) and compare shas with refs/sep27-cli; then spec on.
2. Text-only control on a projector-loaded server.
3. Multi-slot: one image slot + text slots (stall vs tokens); several image slots (scratch/route bugs).
4. --image-max-tokens 1024 as the served default; bottle label at full.
5. Optional: thinking arm on the abstract rows (-n 1500) for drafter acceptance on long image-conditioned text;
   encoder-pooled features as an image-similarity probe (small tool against libmtmd, no hook today).
