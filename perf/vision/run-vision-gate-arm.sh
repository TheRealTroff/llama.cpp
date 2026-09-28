#!/bin/bash
# The VISION ARM of the mint (owner 2026-09-28: "I'll take 1 and 3"). Mirrors perf/run-multislot-gate.sh: the served pick
# with the projector at FIXED depth 1, the depth controller off and streaming top-k off (garbage shows as text, and a
# controller fork cannot masquerade as a defect - mint-controller-arm-is-statistical), a 12-row subset of
# perf/vision/prompts.tsv (facts across rungs, two long answers, the text-only control), then the multi-slot vision
# arm (one image beside three text requests, four images at once). PASS = every sha equals the line's recorded
# reference; a line without a reference prints its shas for this file to record.
#   LINE=q4|ud [TAG=..] [B=tree] [PORT=8095] bash perf/vision/run-vision-gate-arm.sh      VISION=0 on run-prod-pick.sh skips it
set -u
B=${B:-/Users/troff/play/llama.cpp-prod}
LINE=${LINE:-q4}; TAG=${TAG:-vision-arm-$(date +%m%d-%H%M)}; PORT=${PORT:-8095}
OUT=/Users/troff/play/kvquant-experiments/results
MMPROJ=${MMPROJ:-/Users/troff/play/qwen3.8-mmproj-F16.gguf}
[ -f "$MMPROJ" ] || { echo "vision arm: no projector at $MMPROJ - skipped"; exit 0; }
ROWS='^IMG_3428	(1024|full)	table|^IMG_2924	1024	(tonic|sober)|^IMG_3414	1024	name|^gdn	(full	layers|1024	psi)|^rome	(1024	low14|full	storms)|^synth_sign	512	fr|^IMG_3428	1024	meal|^none'
case "$LINE" in   # "image-size-qid:sha12 ..." and "phase/key:sha12 ..." recorded on the merged prod bf92ab864 (2026-09-28,
                  # results vision-arm-rec-*); override from the environment. Fixed depth 1, controller off, top-k stream off.
  q4) REF=${REF_Q4:-"IMG_3414-1024-name:1bfb2d9653f6 IMG_3428-full-table:73445aa688fd IMG_3428-1024-table:73445aa688fd IMG_3428-1024-meal:e57743f68d6a gdn-full-layers:b5d69afccc0d gdn-1024-psi:1ef942d0b0c1 rome-1024-low14:1a252402972f rome-full-storms:ed08e7c3c851 none-none-textonly:d8426862f695 synth_sign-512-fr:7fe30270beee IMG_2924-1024-tonic:12a008e44543 IMG_2924-1024-sober:3f5d2d9b7a65"}
      MSREF=${REF_MS_Q4:-"mixed/txt-chat:264e0e98e975 mixed/img-menu:20cf47c123e7 mixed/txt-code:231240cb6b3e mixed/txt-prose:8fa89a66edd0 quad/img-rome:29d6d85c3d80 quad/img-menu:7dc23a2a2b35 quad/img-steak:7c39e1e2822c quad/img-bar:d16d3feb22cf"} ;;
  ud) REF=${REF_UD:-"IMG_3414-1024-name:1bfb2d9653f6 IMG_3428-full-table:73445aa688fd IMG_3428-1024-table:73445aa688fd IMG_3428-1024-meal:241f11891623 gdn-full-layers:b5d69afccc0d gdn-1024-psi:82c9b37aad92 rome-1024-low14:1a252402972f rome-full-storms:ed08e7c3c851 none-none-textonly:d8426862f695 synth_sign-512-fr:37da550ad4f4 IMG_2924-1024-tonic:59e66134ed0d IMG_2924-1024-sober:67418913369e"}
      MSREF=${REF_MS_UD:-"mixed/txt-chat:71212738614d mixed/txt-code:06c0930c22a9 mixed/img-menu:753ef06178cc mixed/txt-prose:7d678a414d85 quad/img-rome:4127c3564c8b quad/img-menu:753ef06178cc quad/img-steak:5cfc2af73acd quad/img-bar:5d9a98b21c79"} ;;
  *) echo "unknown LINE $LINE"; exit 1 ;;
esac
export B LINE PORT MMPROJ ROWS ARM=spec KV=turbo4 PICK_SPEC_EV=0 PICK_DEPTH=1 EXTRA="GGML_TOPK_STREAM=0" TAG="$TAG-vision"
bash "$B/perf/vision/run-vision-server.sh" > "$OUT/$TAG-vision-$LINE.console.log" 2>&1
D="$OUT/$TAG-$LINE-spec"   # run-vision-server.sh names its dir $TAG-$LINE-$ARM with the TAG exported above
python3 - "$D/summary.tsv" "$REF" <<'PY'
import sys, csv
rows = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
ref = dict(kv.split(':') for kv in sys.argv[2].split() if ':' in kv)
ok = True; out = []
for r in rows:
    key = f"{r['image']}-{r['size']}-{r['qid']}"; exp = ref.get(key)
    st = 'ok' if exp is None or exp == r['sha12'] else 'SHA MOVED'
    if exp is not None and exp != r['sha12']: ok = False
    print(f"  {key:26} sha={r['sha12']} ref={exp or '(unrecorded)':12} {st}")
    out.append(f"{key}:{r['sha12']}")
print('  vision arm (one slot): ' + ('PASS' if ok and ref else ('FAIL' if ref else 'no reference on this line yet - record: ' + ' '.join(out))))
sys.exit(0 if ok else 1)
PY
rc1=$?
grep -q 'could not ingest' "$D/server.log" && { echo "  vision arm: the drafter fell back on an image (spec_off_request) - FAIL"; rc1=1; }
echo
TAG="$TAG-vision-ms" REF_MS="$MSREF" bash "$B/perf/vision/run-vision-multislot.sh" 2>&1 | grep -E '^(mixed|quad|RESULT|MS-REF)' | sed 's/^/  /'
rc2=${PIPESTATUS[0]}
echo "  vision arm: $([ $rc1 = 0 ] && [ $rc2 = 0 ] && echo PASS || echo FAIL) (one-slot rc=$rc1, multi-slot rc=$rc2)"
exit $(( rc1 + rc2 ))
