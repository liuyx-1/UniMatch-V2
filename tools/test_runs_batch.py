"""Batch-test a list of run directories with up to N parallel workers and
aggregate per-run test_metrics.csv into Markdown + LaTeX comparison tables.

Each run dir is expected to contain best.pth. test.py auto-detects VA / edge /
affinity_side from the ckpt's state_dict, so we DON'T pass module switches —
test.py figures out the right forward path itself.

Usage:
    python tools/test_runs_batch.py \\
        --exp-root /root/autodl-tmp/exp/endoscapes_seg50 \\
        --splits-root /root/autodl-tmp/data/autonomous_surgery/splits \\
        --dataset endoscapes_seg50 \\
        --workers 3 \\
        --out-dir /root/autodl-tmp/exp/endoscapes_seg50/_batch_test \\
        --runs - <<EOF
unimatch_v2_base_r0.10_bs16_lr5e-5_cr518_va
unimatch_v2_affinity_min_r0.10_bs16_lr5e-5_cr518_va_joint
...
EOF

Or via file:
    python tools/test_runs_batch.py --runs runs_list.txt ...

Add --skip-existing to reuse existing test_metrics.csv.
"""
from __future__ import annotations
import argparse, csv, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------- TAG parsing ----------------
RATE_RE  = re.compile(r'_r(\d+\.?\d*)')        # _r0.10 or _r0.1
BS_RE    = re.compile(r'_bs(\d+)')
LR_RE    = re.compile(r'_lr([0-9eE\.\-\+]+)')
EP_RE    = re.compile(r'_ep(\d+)')
CR_RE    = re.compile(r'_cr(\d+)')


def parse_tag(dir_name: str) -> Dict[str, Optional[str]]:
    """Pull hyperparameter chunks out of a run-dir name like
       'unimatch_v2_affinity_min_r0.10_bs16_lr5e-5_cr518_va_joint_edge'
    """
    tag = dir_name.replace('unimatch_v2_', '', 1)
    out = {
        'tag':     tag,
        'rate':    None,
        'bs':      None,
        'lr':      None,
        'epochs':  None,
        'crop':    None,
        'has_va':     '_va'    in tag,
        'has_joint':  '_joint' in tag,
        'has_edge':   '_edge'  in tag,
        'has_bnd':    '_bnd'   in tag,
        'has_cma':    '_cma'   in tag,
    }
    m = RATE_RE.search(tag);  out['rate']   = m.group(1) if m else None
    m = BS_RE.search(tag);    out['bs']     = m.group(1) if m else None
    m = LR_RE.search(tag);    out['lr']     = m.group(1) if m else None
    m = EP_RE.search(tag);    out['epochs'] = m.group(1) if m else None
    m = CR_RE.search(tag);    out['crop']   = m.group(1) if m else None

    # Normalise rate: '0.1' → '0.10' for split-dir matching
    if out['rate'] and len(out['rate']) == 3 and out['rate'][1] == '.':
        out['rate'] = out['rate'] + '0'
    return out


def variant_label(tag: str) -> str:
    """Short human-readable label for the table."""
    if tag.startswith('base_'):           base = 'base'
    elif tag.startswith('affinity_min_cma_'): base = 'aff_cma'
    elif tag.startswith('affinity_min_'): base = '+text'
    elif tag.startswith('boundary_'):     base = '+bnd'
    elif tag.startswith('full_'):         base = 'full'
    else:                                 base = tag.split('_')[0]
    flags = []
    if '_va'    in tag: flags.append('VA')
    if '_joint' in tag: flags.append('text')
    if '_edge'  in tag: flags.append('edge')
    if '_bnd'   in tag and not tag.startswith('boundary_'): flags.append('bnd')
    return f'{base}({"+".join(flags)})' if flags else base


# Ablation grouping — drives the section headers in the output tables.
# Order matters: it determines the section order in the markdown / LaTeX output.
GROUPS_ORDER = [
    '① base (VA)',
    '② +text  (VA + LC-PAM)',
    '③ +edge  (VA + EABR)',
    '④ +edge+text  (VA + LC-PAM + EABR)',
    '⑤ +boundary (VA + boundary head)',
    '⑥ full  (VA + text + edge + bnd)',
    'Official / Legacy (no VA)',
    'Legacy CMA / warmstart',
    'Other',
]


def categorize(tag: str) -> str:
    """Return one of GROUPS_ORDER labels."""
    has_va     = '_va'    in tag
    has_joint  = '_joint' in tag
    has_edge   = '_edge'  in tag
    has_bnd    = '_bnd'   in tag
    has_cma    = '_cma'   in tag

    if has_cma:
        return 'Legacy CMA / warmstart'

    if tag.startswith('base_'):
        if not has_va: return 'Official / Legacy (no VA)'
        if has_edge:   return '③ +edge  (VA + EABR)'
        return '① base (VA)'

    if tag.startswith('affinity_min_'):
        if not has_va: return 'Official / Legacy (no VA)'
        if not has_joint:
            return 'Legacy CMA / warmstart'   # affinity_min without joint = warmstart-only
        if has_edge:   return '④ +edge+text  (VA + LC-PAM + EABR)'
        return '② +text  (VA + LC-PAM)'

    if tag.startswith('boundary_'):
        return '⑤ +boundary (VA + boundary head)'

    if tag.startswith('full_'):
        if has_joint and has_edge and has_bnd:
            return '⑥ full  (VA + text + edge + bnd)'
        return '⑥ full  (VA + text + edge + bnd)'

    return 'Other'


# ---------------- test invocation ----------------
def split_dir_for(splits_root: Path, dataset: str, rate: str) -> Path:
    return splits_root / f'unimatch_splits_{dataset}_{rate}_seed42'


def find_test_id(splits_root: Path, dataset: str, rate: str) -> Optional[Path]:
    sd = split_dir_for(splits_root, dataset, rate)
    for n in ('test.txt', 'val.txt'):
        p = sd / n
        if p.exists(): return p
    return None


def run_one_test(run_dir: Path, dataset: str, splits_root: Path,
                 device: str, skip_existing: bool) -> Tuple[str, str]:
    """Returns (status: 'ok'|'skip'|'fail', message)."""
    name = run_dir.name
    info = parse_tag(name)
    csv_path = run_dir / 'test_metrics.csv'

    if skip_existing and csv_path.exists() and csv_path.stat().st_size > 100:
        return ('skip', f'{name}: csv already present')

    ck = run_dir / 'best.pth'
    if not ck.exists():
        return ('skip', f'{name}: no best.pth — skipped')

    rate = info['rate'] or '0.10'
    test_id = find_test_id(splits_root, dataset, rate)
    if test_id is None:
        return ('fail', f'{name}: no test/val split at rate={rate}')

    cfg = Path('configs') / f'{dataset}.yaml'
    train_id = split_dir_for(splits_root, dataset, rate) / 'labeled.txt'
    cmd = [
        'python', 'test.py',
        '--config',         str(cfg),
        '--checkpoint',     str(ck),
        '--id-path',        str(test_id),
        '--train-id-path',  str(train_id),
        '--train-freq-cache', str(run_dir / 'train_pixel_freq_test.json'),
        '--use-ema', '--ece', '--ap',
        '--save-csv',       str(csv_path),
        '--device',         device,
    ]
    log = run_dir / 'test_batch.log'
    try:
        with open(log, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as e:
        return ('fail', f'{name}: test.py exit {e.returncode}, see {log}')
    return ('ok', f'{name}: wrote {csv_path.name}')


# ---------------- aggregate ----------------
SUMMARY_KEYS = [
    ('miou_all', 'mIoU'),
    ('mdice_all', 'mDice'),
    ('mAP_present', 'mAP'),
    ('pa', 'PA'),
    ('fwiou', 'fwIoU'),
]
BIAS_KEYS = [
    ('mIoU_head', 'mIoU_h'),
    ('mIoU_body', 'mIoU_b'),
    ('mIoU_tail', 'mIoU_t'),
    ('gap_HT', 'gap_HT'),
]


def parse_csv(p: Path) -> Dict[str, Dict[str, float]]:
    if not p.exists(): return {}
    summary, bias = {}, {}
    with open(p) as f:
        rd = csv.reader(f); next(rd, None)
        for row in rd:
            if not row: continue
            tag = row[0].strip()
            if tag == 'summary' and len(row) >= 3:
                try: summary[row[1]] = float(row[2])
                except: pass
            elif tag == 'bias' and len(row) >= 3:
                try: bias[row[1]] = float(row[2])
                except: pass
    return {'summary': summary, 'bias': bias}


def emit_tables(rows: List[Dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cols  = ['run', 'variant', 'rate', 'bs', 'lr', 'crop', 'ep']
    cols += [h[1] for h in SUMMARY_KEYS] + [h[1] for h in BIAS_KEYS]

    # ---------- group by ablation category ----------
    grouped: Dict[str, List[Dict]] = {g: [] for g in GROUPS_ORDER}
    for r in rows:
        grouped.setdefault(r.get('group', 'Other'), []).append(r)

    def lr_sort_key(r):
        try: return float(r.get('lr', 'inf'))
        except: return float('inf')
    for g in grouped:
        grouped[g].sort(key=lambda r: (lr_sort_key(r), r.get('bs', '')))

    ordered_groups = [g for g in GROUPS_ORDER if grouped.get(g)]
    other_groups   = [g for g in grouped if g not in GROUPS_ORDER and grouped[g]]
    ordered_groups += other_groups

    # ---------- markdown ----------
    md = []
    for g in ordered_groups:
        gr = grouped[g]
        if not gr: continue
        md.append(f'\n### {g}   ({len(gr)} run{"s" if len(gr)!=1 else ""})\n')
        md.append('| ' + ' | '.join(cols) + ' |')
        md.append('|' + '|'.join(['---'] * len(cols)) + '|')
        for r in gr:
            md.append('| ' + ' | '.join(str(r.get(c, '—')) for c in cols) + ' |')
    (out_dir / 'batch_table.md').write_text('\n'.join(md), encoding='utf-8')

    # ---------- csv (single flat sheet, with group column) ----------
    with open(out_dir / 'batch_table.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['group'] + cols)
        for g in ordered_groups:
            for r in grouped[g]:
                w.writerow([g] + [r.get(c, '') for c in cols])

    # ---------- latex (one \multicolumn header per group) ----------
    align = 'l l c c c c c' + ' c' * (len(cols) - 7)
    safe  = lambda s: str(s).replace('_', '\\_').replace('%', '\\%')
    parts = [
        '\\begin{table*}[t]\n\\centering\\small\n'
        '\\setlength{\\tabcolsep}{4pt}\n'
        '\\renewcommand{\\arraystretch}{1.12}\n'
        f'\\caption{{Batch test of {len(rows)} runs, grouped by ablation category.}}\n'
        f'\\label{{tab:batch_runs}}\n'
        f'\\begin{{tabular}}{{{align}}}\n\\toprule\n'
        + ' & '.join(f'\\textbf{{{safe(c)}}}' for c in cols) + ' \\\\'
    ]
    for g in ordered_groups:
        gr = grouped[g]
        if not gr: continue
        parts.append(f'\n\\midrule\n\\multicolumn{{{len(cols)}}}{{l}}'
                     f'{{\\textit{{{safe(g)}}}}} \\\\')
        for r in gr:
            parts.append('\n' + ' & '.join(safe(r.get(c, '—')) for c in cols) + ' \\\\')
    parts.append('\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n')
    (out_dir / 'batch_table.tex').write_text(''.join(parts), encoding='utf-8')

    # ---------- console summary ----------
    print()
    for g in ordered_groups:
        n = len(grouped[g])
        if n: print(f'  [{g:<48s}]  {n} run(s)')
    print(f'\n[saved] {out_dir / "batch_table.md"}')
    print(f'[saved] {out_dir / "batch_table.csv"}')
    print(f'[saved] {out_dir / "batch_table.tex"}')


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp-root',     type=Path, required=True,
                    help='dir containing the unimatch_v2_* run subdirs')
    ap.add_argument('--splits-root',  type=Path, required=True)
    ap.add_argument('--dataset',      type=str, default='endoscapes_seg50')
    ap.add_argument('--runs',         type=str, default='-',
                    help='file with one run-dir name per line; "-" for stdin')
    ap.add_argument('--workers',      type=int, default=3)
    ap.add_argument('--device',       type=str, default='cuda:0')
    ap.add_argument('--skip-existing', action='store_true')
    ap.add_argument('--skip-smoke',    action='store_true',
                    help='skip dirs containing _ep1')
    ap.add_argument('--out-dir',       type=Path, required=True)
    args = ap.parse_args()

    if args.runs == '-':
        lines = sys.stdin.read().splitlines()
    else:
        lines = Path(args.runs).read_text().splitlines()
    names = [ln.strip() for ln in lines if ln.strip() and not ln.startswith('#')]
    if args.skip_smoke:
        names = [n for n in names if '_ep1_' not in n and not n.endswith('_ep1')]
    run_dirs_all = [args.exp_root / n for n in names]

    # Pre-filter: drop entries with no best.pth so they don't even spawn workers
    run_dirs, missing_ckpt = [], []
    for rd in run_dirs_all:
        (run_dirs if (rd / 'best.pth').exists() else missing_ckpt).append(rd)
    if missing_ckpt:
        print(f'[skip-pre] {len(missing_ckpt)} runs without best.pth:')
        for rd in missing_ckpt:
            print(f'   · {rd.name}')
    print(f'[plan] {len(run_dirs)} runs to test · {args.workers} parallel workers')

    # ---- run tests ----
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one_test, rd, args.dataset, args.splits_root,
                          args.device, args.skip_existing): rd for rd in run_dirs}
        for fut in as_completed(futs):
            status, msg = fut.result()
            tag = {'ok': '✓', 'skip': '·', 'fail': '✗'}[status]
            print(f'  [{tag}] {msg}')
            results.append((status, msg))

    n_ok = sum(1 for s,_ in results if s != 'fail')
    print(f'\n[results] {n_ok}/{len(results)} ok or skipped')

    # ---- aggregate ----
    rows = []
    for rd in run_dirs:
        info = parse_tag(rd.name)
        m = parse_csv(rd / 'test_metrics.csv')
        if not m: continue
        tag = rd.name.replace('unimatch_v2_', '')
        row = {
            'run':     tag,
            'variant': variant_label(tag),
            'group':   categorize(tag),
            'rate':    info['rate']   or '',
            'bs':      info['bs']     or '',
            'lr':      info['lr']     or '',
            'crop':    info['crop']   or '',
            'ep':      info['epochs'] or '',
        }
        for k, label in SUMMARY_KEYS:
            v = m['summary'].get(k)
            row[label] = f'{v*100:.2f}' if isinstance(v, float) else '—'
        for k, label in BIAS_KEYS:
            v = m['bias'].get(k)
            row[label] = f'{v*100:.2f}' if isinstance(v, float) else '—'
        rows.append(row)
    emit_tables(rows, args.out_dir)


if __name__ == '__main__':
    main()
