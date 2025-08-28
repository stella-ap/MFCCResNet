import h5py, numpy as np, pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# ------------------------------
# Config: normal + stego files
# ------------------------------
FILES = {
    256: {
        "normal": "/input/mfcc-mdct-windows-features/normal_features_MFCC_win256.h5",
        "stego":  "/kaggle/input/mfcc-mdct-windows-features/stego_features_MFCC_win256.h5"
    },
    512: {
        "normal": "/input/mfcc-mdct-windows-features/normal_features_MFCC_win512.h5",
        "stego":  "/input/mfcc-mdct-windows-features/stego_features_MFCC_win512.h5"
    },
    1024: {
        "normal": "/input/mfcc-mdct-windows-features/normal_features_MFCC_win1024.h5",
        "stego":  "/input/mfcc-mdct-windows-features/stego_features_MFCC_win1024.h5"
    }
}

# ------------------------------
def load_h5(path):
    with h5py.File(path, "r") as f:
        key = list(f.keys())[0]   # 'X'
        return f[key][:]

def aggregate_features(mfcc):
    return np.mean(mfcc, axis=1)  # (samples, 13)

def single_feature_auc(X, y, splits=5):
    kf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    aucs = []
    for tr, te in kf.split(X, y):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[tr].reshape(-1,1), y[tr])
        y_prob = clf.predict_proba(X[te].reshape(-1,1))[:,1]
        aucs.append(roc_auc_score(y[te], y_prob))
    return float(np.mean(aucs))

def analyze_dataset(normal, stego):
    # build dataset
    Xn, Xs = aggregate_features(normal), aggregate_features(stego)
    X = np.vstack([Xn, Xs])
    y = np.hstack([np.zeros(len(Xn)), np.ones(len(Xs))])

    # sanity
    stats = {
        "mean": X.mean(axis=0),
        "variance": X.var(axis=0),
        "std": X.std(axis=0)
    }
    df = pd.DataFrame(stats, index=[f"MFCC{i}" for i in range(X.shape[1])])

    # discriminative tests
    F, p = f_classif(X, y)
    MI = mutual_info_classif(X, y, random_state=42)
    aucs = [single_feature_auc(X[:,i], y) for i in range(X.shape[1])]

    df["F_score"] = F
    df["p_value"] = p
    df["Mutual_Info"] = MI
    df["ROC_AUC"] = aucs
    return df

# ------------------------------
all_reports = {}
for N, paths in FILES.items():
    normal = load_h5(paths["normal"])
    stego  = load_h5(paths["stego"])
    report = analyze_dataset(normal, stego)
    all_reports[N] = report
    print(f"\n===== Window N={N} =====")
    print(report)

# Combine
combined = pd.concat(all_reports, axis=1)
combined.to_csv("mfcc_bin_comparison_normal_vs_stego.csv")
print("\n Results saved to mfcc_bin_comparison_normal_vs_stego.csv")
