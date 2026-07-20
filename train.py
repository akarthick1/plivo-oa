import argparse
import csv
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

from features import load_wav, speech_before, frame_energy_db, f0_contour

def extract_features(x, sr, pause_start):
    total_duration = pause_start
    seg = speech_before(x, sr, pause_start, window_s=1.5)
    
    if len(seg) < sr // 10:
        return np.zeros(6, dtype=np.float32)
        
    e = frame_energy_db(seg, sr)
    f0 = f0_contour(seg, sr)
    voiced = f0[f0 > 0]
    
    e_mean = e.mean() if len(e) > 0 else 0.0
    e_slope = np.polyfit(np.arange(len(e)), e, 1)[0] if len(e) > 1 else 0.0
    
    p_mean = voiced.mean() if len(voiced) > 0 else 0.0
    p_slope = np.polyfit(np.arange(len(voiced)), voiced, 1)[0] if len(voiced) > 1 else 0.0
    
    return np.array([
        total_duration, e_mean, e_slope, p_mean, p_slope, len(seg) / sr 
    ], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    # Accept multiple directories using nargs="+"
    ap.add_argument("--data_dirs", nargs="+", required=True) 
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    cache = {}
    X, y, groups, keys = [], [], [], []
    
    for data_dir in args.data_dirs:
        labels_path = os.path.join(data_dir, "labels.csv")
        rows = list(csv.DictReader(open(labels_path)))
        dataset_name = os.path.basename(os.path.normpath(data_dir))
        
        for r in rows:
            path = os.path.join(data_dir, r["audio_file"])
            if path not in cache:
                cache[path] = load_wav(path)
            x, sr = cache[path]
            
            # Make turn_id unique across datasets
            unique_turn_id = f"{dataset_name}_{r['turn_id']}"
            
            X.append(extract_features(x, sr, float(r["pause_start"])))
            y.append(1 if r["label"] == "eot" else 0)
            groups.append(unique_turn_id)
            keys.append((unique_turn_id, r["pause_index"]))
            
    X, y = np.array(X), np.array(y)

    feature_cols = ['total_duration', 'e_mean', 'e_slope', 'p_mean', 'p_slope', 'window_duration']
    df_features = pd.DataFrame(X, columns=feature_cols)
    df_features['target'] = y
    df_features['turn_id'] = [k[0] for k in keys]
    df_features['pause_index'] = [k[1] for k in keys]
    
    os.makedirs('outputs', exist_ok=True)
    df_features.to_csv('outputs/feature_matrix_combined.csv', index=False)
    print(f"Combined feature matrix saved -> shape {df_features.shape}")

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X, y, groups))
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42)
    
    clf.fit(X[tr], y[tr])
    print(f"Held-out turn accuracy: {clf.score(X[te], y[te]):.3f} (chance ~ {max(np.mean(y), 1-np.mean(y)):.3f})")

    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    
    with open('outputs/model_combined.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print("Saved trained model to outputs/model_combined.pkl")
    
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pi_p in zip(keys, p):
            w.writerow([tid, pi, f"{pi_p:.4f}"])

if __name__ == "__main__":
    main()