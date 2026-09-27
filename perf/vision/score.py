#!/usr/bin/env python3
"""Score a vision run (exp/vision-gate). No semantic judging anywhere:
  fact     every ';'-separated expect string appears in the answer (case-insensitive)
  edit     difflib similarity ratio of the answer vs expect (whitespace/case-normalized), pass >= 0.95
  abstract each ';' group satisfied by any of its '|' alternatives (a checklist the owner approves once)
  desc     no expectation; only the sha is meaningful (fork vs reference)
  --ref DIR  also compare each row's sha12 with the same row in a reference run (byte-equality gate)
usage: score.py <results dir> [--ref <reference results dir>] [--prompts prompts.tsv]
"""
import sys, os, csv, difflib, re, argparse

def norm(s):
    return re.sub(r'\s+', ' ', s.strip()).lower()

def load_summary(d):
    p = os.path.join(d, 'summary.tsv')
    with open(p) as f:
        return {(r['image'], r['size'], r['qid']): r for r in csv.DictReader(f, delimiter='\t')}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir'); ap.add_argument('--ref'); ap.add_argument('--prompts', default=os.path.join(os.path.dirname(__file__), 'prompts.tsv'))
    a = ap.parse_args()
    rows = {}
    for line in open(a.prompts):
        if line.startswith('#') or not line.strip(): continue
        image, size, qid, kind, n, prompt, expect = (line.rstrip('\n').split('\t') + [''])[:7]
        rows[(image, size, qid)] = (kind, expect)
    got = load_summary(a.dir)
    ref = load_summary(a.ref) if a.ref else {}
    npass = nfail = 0
    print(f"{'image':9} {'size':5} {'qid':9} {'kind':8} {'enc_ms':>6} {'wall':>5} {'score':>7}  {'sha':12} {'ref':5}  answer")
    for key, r in got.items():
        kind, expect = rows.get(key, ('?', ''))
        ans = open(os.path.join(a.dir, r['file'])).read()
        score = ''
        if kind == 'fact':
            ok = all(norm(e) in norm(ans) for e in expect.split(';')); score = 'PASS' if ok else 'FAIL'
        elif kind == 'edit':
            ratio = difflib.SequenceMatcher(None, norm(ans), norm(expect)).ratio(); score = f'{ratio:.3f}' + ('' if ratio >= 0.95 else '!')
            ok = ratio >= 0.95
        elif kind == 'abstract':
            groups = [g.split('|') for g in expect.split(';') if g]
            hits = [any(norm(alt) in norm(ans) for alt in g) for g in groups]
            ok = all(hits); score = f'{sum(hits)}/{len(hits)}' + ('' if ok else '!')
        else:
            ok = None; score = '-'
        if ok is True: npass += 1
        if ok is False: nfail += 1
        rs = ''
        if ref:
            rr = ref.get(key); rs = '-' if rr is None else ('same' if rr['sha12'] == r['sha12'] else 'DIFF')
        first = ans.strip().splitlines()[0][:70] if ans.strip() else ''
        print(f"{key[0]:9} {key[1]:5} {key[2]:9} {kind:8} {r['encode_ms']:>6} {r['wall_s']:>5} {score:>7}  {r['sha12']:12} {rs:5}  {first}")
    print(f"\nscored: {npass} pass, {nfail} fail" + (f"; sha vs ref: {sum(1 for k,r in got.items() if k in ref and ref[k]['sha12']==r['sha12'])}/{sum(1 for k in got if k in ref)} same" if ref else ''))

if __name__ == '__main__':
    main()
