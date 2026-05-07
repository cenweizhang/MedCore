"""Merge abl1 worker JSON files into the main cascade_results_v8.json."""
import json, os, sys

base = "results/ablation_v8/abl1_no_boundary_fisher"
main_path = os.path.join(base, "cascade_results_v8.json")

with open(main_path) as f:
    main = json.load(f)

existing_keys = {
    (r.get("head_sp_target"), r.get("mlp_sp_target"), r.get("phase"))
    for r in main["cascade_results"]
}

merged = 0
for idx in [1, 2, 3]:
    wpath = os.path.join(base, f"cascade_results_v8_w0{idx}.json")
    if not os.path.exists(wpath):
        print(f"  MISSING: {wpath}")
        continue
    with open(wpath) as f:
        w = json.load(f)
    for r in w["cascade_results"]:
        key = (r.get("head_sp_target"), r.get("mlp_sp_target"), r.get("phase"))
        if key not in existing_keys:
            main["cascade_results"].append(r)
            existing_keys.add(key)
            merged += 1
            print(f"  added  h={r.get('head_sp_target',0):.2f} m={r.get('mlp_sp_target',0):.2f} phase={r.get('phase')}")
        else:
            print(f"  skip (duplicate)  {key}")

with open(main_path, "w") as f:
    json.dump(main, f, indent=2)
print(f"\nMerged {merged} new rows → {main_path}")
print(f"Total cascade rows: {len(main['cascade_results'])}")
