import argparse
import csv
import os
import pickle
import numpy as np

from features import load_wav
from train import extract_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default="outputs/model_combined.pkl")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found at {args.model}. Train first.")
    
    with open(args.model, 'rb') as f:
        clf = pickle.load(f)

    labels_path = os.path.join(args.data_dir, "labels.csv")
    rows = list(csv.DictReader(open(labels_path)))
    
    cache = {}
    predictions = []

    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        
        pause_start = float(r["pause_start"])
        feat = extract_features(x, sr, pause_start)
        
        feat_matrix = feat.reshape(1, -1) 
        p_eot = clf.predict_proba(feat_matrix)[0, 1]
        
        predictions.append({
            "turn_id": r["turn_id"],
            "pause_index": r["pause_index"],
            "p_eot": f"{p_eot:.4f}"
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["turn_id", "pause_index", "p_eot"])
        w.writeheader()
        w.writerows(predictions)
        
    print(f"Wrote {len(predictions)} predictions to {args.out}")

if __name__ == "__main__":
    main()