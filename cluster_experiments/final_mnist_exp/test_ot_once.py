# test_ot_once.py
import json, os, argparse, importlib.util, sys
import matplotlib
matplotlib.use("Agg")

import utils

def import_exp_module(path):
    spec = importlib.util.spec_from_file_location("expmod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["expmod"] = mod
    spec.loader.exec_module(mod)
    return mod


ap = argparse.ArgumentParser()
ap.add_argument("--exp_file", required=True, help="Path to data_transform.py")
ap.add_argument("--data1", required=True, help="Dataset key for source (e.g. mnist_unrotated)")
ap.add_argument("--data2", required=True, help="Dataset key for target (e.g. mnist_rotated)")
ap.add_argument("--feature", required=True, choices=["xs", "bs"])
ap.add_argument("--method", default="sinkhorn",
                help="sinkhorn | sinkhorn_stabilized | sinkhorn_log | greenkhorn | emd")
ap.add_argument("--reg", default="0.1", help='Regularization (float) or "None" for exact EMD')
ap.add_argument("--out", required=True)
args = ap.parse_args()

exp = import_exp_module(args.exp_file)

d1 = exp.build_dataset_by_key(args.data1)
d2 = exp.build_dataset_by_key(args.data2)

reg_val = None if str(args.reg).lower() == "none" else float(args.reg)
method = "emd" if reg_val is None else args.method

metrics = utils.test_ot_once(
    d1, d2,
    feature=args.feature,
    method=method,
    reg=reg_val,
    is_verbose=True,
    return_confmat=True
)

payload = {
    "exp_file": args.exp_file,
    "data1": args.data1,
    "data2": args.data2,
    "feature": args.feature,
    "method": method,
    "reg": None if reg_val is None else float(reg_val),
    "metrics": metrics
}

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(payload, f, indent=2)

print(f"Wrote {args.out}")
