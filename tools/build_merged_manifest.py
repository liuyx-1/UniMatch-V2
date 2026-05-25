"""Build merged manifest across 5 datasets with synonym-merged global class space.

Background classes are EXCLUDED (treated as ignore / no class).
Synonyms are merged:
    shaft               (2017_parts)  =  instrument-shaft     (2018)
    wrist               (2017_parts)  =  instrument-wrist     (2018)
    clasper             (2017_parts)  =  instrument-clasper   (2018)
    gallbladder         (endoscapes)   =  gallbladder          (cholecseg8k)
    cystic_duct         (endoscapes)   =  cystic_duct          (cholecseg8k)
    Ultrasound Probe    (2017_type)    =  ultrasound-probe     (2018)

Outputs:
    manifest_merged.jsonl      one line per image with:
        image, mask, dataset, classes_global, classes_orig, has_rare_class, sample_weight
    global_classes.json        {
        names: [33 class names],
        remap: { ds: {orig_id: global_id_or_null_for_bg} },
        valid_global: { ds: [list of global ids this ds annotates] },
        original_names: { ds: [orig class names] }
    }

Usage:
    python tools/build_merged_manifest.py \\
        --manifest-dir /data/pretrained/siglip_train \\
        --out-dir      /data/pretrained/siglip_train/merged
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np


# Per-dataset original class names (kept identical to build_siglip_manifest.py)
ORIGINAL = {
    'endoscapes_seg50': ['background-tissue', 'cystic_plate', 'calot_triangle',
                          'cystic_artery', 'cystic_duct', 'gallbladder', 'tool'],
    'cholecseg8k': ['background', 'abdominal_wall', 'liver', 'gastrointestinal_tract',
                     'fat', 'grasper', 'connective_tissue', 'blood', 'cystic_duct',
                     'l_hook_electrocautery', 'gallbladder', 'hepatic_vein', 'liver_ligament'],
    'endovis2018': ['background-tissue', 'instrument-shaft', 'instrument-clasper',
                     'instrument-wrist', 'kidney-parenchyma', 'covered-kidney',
                     'thread', 'clamps', 'suturing-needle', 'suction-instrument',
                     'small-intestine', 'ultrasound-probe'],
    'endovis2017_parts': ['background', 'shaft', 'wrist', 'clasper'],
    'endovis2017_type': ['background', 'Bipolar Forceps', 'Prograsp Forceps',
                          'Large Needle Driver', 'Vessel Sealer', 'Grasping Retractor',
                          'Monopolar Curved Scissors', 'Ultrasound Probe'],
}

# Background tokens (mapped to None = excluded)
BG_SET = {'background', 'background-tissue'}

# Synonym normalisation: orig_name (lowercased) -> canonical_name
SYNONYMS = {
    'instrument-shaft':   'shaft',
    'instrument-wrist':   'wrist',
    'instrument-clasper': 'clasper',
    'ultrasound probe':   'ultrasound_probe',
    'ultrasound-probe':   'ultrasound_probe',
}


def _canon(name: str) -> str:
    n = name.strip().lower()
    if n in SYNONYMS: return SYNONYMS[n]
    return n.replace(' ', '_').replace('-', '_')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest-dir', type=Path, required=True)
    ap.add_argument('--out-dir',      type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build global class list (skipping bg) ----
    canon_order = []   # canonical name in first-seen order
    canon_seen  = set()
    remap = {}         # { ds: { orig_id: global_id_or_None } }
    valid_global = {}  # { ds: set(global ids) }
    for ds, names in ORIGINAL.items():
        remap[ds] = {}
        valid_global[ds] = set()
        for oid, oname in enumerate(names):
            if oname in BG_SET:
                remap[ds][oid] = None
                continue
            cname = _canon(oname)
            if cname not in canon_seen:
                canon_seen.add(cname)
                canon_order.append(cname)
            gid = canon_order.index(cname)
            remap[ds][oid] = gid
            valid_global[ds].add(gid)

    print(f'[global classes] {len(canon_order)} unique non-bg classes')
    for i, n in enumerate(canon_order):
        owners = [ds for ds, v in valid_global.items() if i in v]
        merged_flag = ' [merged]' if len(owners) > 1 else ''
        print(f'  {i:>2}  {n:<28}  <- {owners}{merged_flag}')

    # ---- Compute global class frequencies (for rare flag + sample_weight) ----
    global_freq = np.zeros(len(canon_order), dtype=np.int64)
    per_ds_records = {}
    for ds in ORIGINAL.keys():
        m = args.manifest_dir / f'manifest_{ds}.jsonl'
        if not m.exists():
            print(f'[skip] {ds}: no manifest at {m}')
            continue
        recs = [json.loads(ln) for ln in m.read_text(encoding='utf-8').splitlines() if ln.strip()]
        per_ds_records[ds] = recs
        for r in recs:
            for c in r.get('classes', []):
                gid = remap[ds].get(int(c))
                if gid is not None:
                    global_freq[gid] += 1
        print(f'  loaded {len(recs):>5} records from {ds}')

    # Rare = bottom 25% by frequency among annotated
    rare_set = set()
    if (global_freq > 0).any():
        nz = global_freq[global_freq > 0]
        thr = np.quantile(nz, 0.25)
        rare_set = set(int(i) for i, f in enumerate(global_freq) if 0 < f <= thr)
    print(f'\n[rare global classes (<= 25%ile freq)] {len(rare_set)}: '
          f'{[canon_order[i] for i in sorted(rare_set)]}')

    # ---- Write merged manifest ----
    out_jsonl = args.out_dir / 'manifest_merged.jsonl'
    n_total = 0; n_rare = 0
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for ds, recs in per_ds_records.items():
            for r in recs:
                gids = sorted({remap[ds][int(c)] for c in r.get('classes', [])
                                if remap[ds][int(c)] is not None})
                if not gids:
                    continue  # all bg, skip
                has_rare = any(g in rare_set for g in gids)
                # sample_weight: harmonic of present-class freqs (rare-class boost)
                inv = sum(1.0 / max(int(global_freq[g]), 1) for g in gids) / len(gids)
                sw = float(inv ** 0.5)
                out = {
                    'image': r['image'],
                    'mask':  r['mask'],
                    'dataset': ds,
                    'classes_global': gids,
                    'classes_orig':   [int(c) for c in r.get('classes', [])],
                    'has_rare_class': has_rare,
                    'sample_weight':  sw,
                }
                f.write(json.dumps(out) + '\n')
                n_total += 1
                if has_rare: n_rare += 1

    info = {
        'names':          canon_order,
        'remap':          {ds: {str(k): v for k, v in m.items()} for ds, m in remap.items()},
        'valid_global':   {ds: sorted(v) for ds, v in valid_global.items()},
        'original_names': ORIGINAL,
        'global_freq':    global_freq.tolist(),
        'rare_global':    sorted(rare_set),
    }
    with open(args.out_dir / 'global_classes.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)

    print(f'\n[write] {out_jsonl}  ({n_total} records, {n_rare} with rare class)')
    print(f'[write] {args.out_dir / "global_classes.json"}')


if __name__ == '__main__':
    main()
