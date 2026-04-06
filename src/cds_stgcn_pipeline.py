"""
STGCN pipeline for CDS spread forecasting and backtesting.

This script loads panel CDS data and supply chain adjacency information,
trains baseline and graph-based forecasting models, evaluates predictive
performance, and runs a protection-based long-short backtest.
"""


from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import random

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


MODE = "DELTA"   # "LEVEL" or "DELTA"
N_HIS = 7
TOP_QUANTILE = 0.2
DV01_PER_BP = 100.0  # cash PnL scale per bp (portfolio scale)
INIT_CASH = 100_000
FEE = 0.0


BATCH_SIZE = 64
EPOCHS = 100
LR = 3e-3
WEIGHT_DECAY = 1e-4
GCN_HIDDEN = 32
TEMPORAL_HIDDEN = 32
PATIENCE = 10
MIN_DELTA = 0.0
DEVICE_STR = "auto"  # "cpu", "cuda", "auto"



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)



PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_SRC = PROJECT_ROOT / "outputs_cds" / "data" / "top50"
VEL_FILE = DATA_SRC / "ve1.csv"
ADJ_FILE = DATA_SRC / "adj.npz"





def load_panel(mode="LEVEL"):
    """
    Returns:
      y_panel: (T, N) if LEVEL, (T-1, N) if DELTA
      colnames: numeric column names
      level_full: (T_level, N) original level series ALWAYS returned for alignment/backtest if needed
    """
    if not VEL_FILE.exists():
        raise FileNotFoundError(f"ve1.csv not found: {VEL_FILE}")

    df = pd.read_csv(VEL_FILE)
    df_num = df.select_dtypes(include=[np.number])

    if df_num.empty:
        raise ValueError("No numeric columns found in ve1.csv")

    df_num = df_num.replace([np.inf, -np.inf], np.nan)
    df_num = df_num.ffill().bfill().fillna(0.0)

    level_full = df_num.to_numpy(dtype=np.float32)
    level_full = np.nan_to_num(level_full, nan=0.0, posinf=0.0, neginf=0.0)

    mode = mode.upper()
    if mode == "LEVEL":
        panel = level_full
    elif mode == "DELTA":
        panel = np.diff(level_full, axis=0)  # (T-1,N)
    else:
        raise ValueError("mode must be LEVEL or DELTA")

    print(f"[data] Panel shape: {panel.shape} | mode={mode}")
    return panel, df_num.columns.tolist(), level_full


def load_adjacency():
    if not ADJ_FILE.exists():
        raise FileNotFoundError(f"adj.npz not found: {ADJ_FILE}")

    adj_sp = sp.load_npz(ADJ_FILE)
    adj = adj_sp.toarray().astype(np.float32)

    # symmetric normalization A_hat = D^{-1/2}(A+I)D^{-1/2}
    N = adj.shape[0]
    I = np.eye(N, dtype=np.float32)
    A_tilde = adj + I
    d = A_tilde.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d + 1e-8)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    print(f"[data] Adjacency shape: {A_norm.shape}")
    return A_norm


def create_sliding_windows(data, n_his=7, n_pred=1):
    """
    data: (T, N)
    X: (B, n_his, N)
    y: (B, N) (n_pred=1)
    """
    T, N = data.shape
    X_list, y_list = [], []
    for t in range(n_his, T - n_pred + 1):
        X_list.append(data[t - n_his:t, :])
        y_list.append(data[t + n_pred - 1, :])
    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    return X, y


def split_80_10_10(T):
    n_train = int(0.8 * T)
    n_val = int(0.1 * T)
    n_test = T - n_train - n_val
    return n_train, n_val, n_test



def compute_metrics_scaled_and_orig(y_true_scaled, y_pred_scaled, scaler=None):
    mse_s = mean_squared_error(y_true_scaled.flatten(), y_pred_scaled.flatten())
    rmse_s = float(np.sqrt(mse_s))
    r2_s = r2_score(y_true_scaled.flatten(), y_pred_scaled.flatten())

    if scaler is None:
        # already original
        y_true_orig = y_true_scaled
        y_pred_orig = y_pred_scaled
    else:
        y_true_orig = scaler.inverse_transform(y_true_scaled)
        y_pred_orig = scaler.inverse_transform(y_pred_scaled)

    mse_o = mean_squared_error(y_true_orig.flatten(), y_pred_orig.flatten())
    rmse_o = float(np.sqrt(mse_o))
    r2_o = r2_score(y_true_orig.flatten(), y_pred_orig.flatten())

    return {
        "mse_s": float(mse_s),
        "rmse_s": float(rmse_s),
        "r2_s": float(r2_s),
        "mse_o": float(mse_o),
        "rmse_o": float(rmse_o),
        "r2_o": float(r2_o),
        "y_true_orig": y_true_orig,
        "y_pred_orig": y_pred_orig
    }


def save_preds_csv(y_true_orig, y_pred_orig, colnames, out_dir, prefix):
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(y_true_orig, columns=colnames).to_csv(out_dir / f"{prefix}_test_y_true.csv", index=False)
    pd.DataFrame(y_pred_orig, columns=colnames).to_csv(out_dir / f"{prefix}_test_y_pred.csv", index=False)


def baseline_naive_level(train_scaled, test_scaled, n_his):
    """
    Predict next level = last observed level in the window (random walk / persistence).
    """
    X_test, y_test = create_sliding_windows(test_scaled, n_his=n_his, n_pred=1)
    yhat = X_test[:, -1, :].copy()  # last level
    return y_test, yhat


def baseline_naive_delta(train_scaled, test_scaled, n_his):
    """
    Predict next delta = 0 (no change).
    """
    X_test, y_test = create_sliding_windows(test_scaled, n_his=n_his, n_pred=1)
    yhat = np.zeros_like(y_test)
    return y_test, yhat


def baseline_ar1(train_scaled, test_scaled, n_his):
    """
    Firm-level AR(1): y_t = a + b y_{t-1} fitted on train_scaled.
    Prediction uses last observed value in window as y_{t-1}.
    """
    y = train_scaled[1:, :]
    x = train_scaled[:-1, :]

    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    cov_xy = ((x - x_mean) * (y - y_mean)).mean(axis=0)
    var_x = ((x - x_mean) ** 2).mean(axis=0) + 1e-8

    b = cov_xy / var_x
    a = y_mean - b * x_mean

    X_test, y_test = create_sliding_windows(test_scaled, n_his=n_his, n_pred=1)
    x_last = X_test[:, -1, :]
    yhat = a[None, :] + b[None, :] * x_last
    return y_test, yhat



class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, A_norm):
        super().__init__()
        self.register_buffer("A_norm", torch.tensor(A_norm, dtype=torch.float32))
        self.theta = nn.Linear(in_channels, out_channels)

    def forward(self, X):
        # X: (B, N, F)
        AX = torch.einsum("ij,bjf->bif", self.A_norm, X)
        return self.theta(AX)


class SimpleSTGCN(nn.Module):
    def __init__(self, num_nodes, n_his, gcn_hidden=32, temporal_hidden=32, A_norm=None):
        super().__init__()
        self.n_his = n_his
        self.gcn = GCNLayer(in_channels=1, out_channels=gcn_hidden, A_norm=A_norm)

        self.temporal_conv = nn.Conv2d(
            in_channels=gcn_hidden,
            out_channels=temporal_hidden,
            kernel_size=(1, 3),
            padding=(0, 1),
        )
        self.fc = nn.Linear(temporal_hidden, 1)

    def forward(self, X):
        # X: (B, T, N)
        B, T, N = X.shape
        assert T == self.n_his

        Xr = X.reshape(B * T, N, 1)       # (B*T,N,1)
        g = self.gcn(Xr)                  # (B*T,N,H)
        g = g.reshape(B, T, N, -1)        # (B,T,N,H)
        g = g.permute(0, 3, 2, 1)         # (B,H,N,T)

        h = self.temporal_conv(g)         # (B,H2,N,T)
        last = h[:, :, :, -1]             # (B,H2,N)
        last = last.permute(0, 2, 1)      # (B,N,H2)

        out = self.fc(last).squeeze(-1)   # (B,N)
        return out


def train_stgcn(panel, A_norm, n_his, device_str="auto",
                batch_size=64, epochs=100, lr=3e-3, weight_decay=1e-4,
                gcn_hidden=32, temporal_hidden=32,
                patience=10, min_delta=0.0,
                out_dir=None, model_name="stgcn_best_model.pt"):
    """
    Train STGCN on given panel (LEVEL or DELTA), using StandardScaler fit on TRAIN split.
    Returns: dict with scaler, y_test_scaled, y_pred_scaled, etc.
    """
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_str == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T, N = panel.shape
    n_train, n_val, n_test = split_80_10_10(T)
    print(f"[split] T={T}, n_train={n_train}, n_val={n_val}, n_test={n_test}")

    train_raw = panel[:n_train]
    val_raw = panel[n_train:n_train + n_val]
    test_raw = panel[n_train + n_val:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    train_scaled = np.nan_to_num(train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    val_scaled = np.nan_to_num(val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    test_scaled = np.nan_to_num(test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    X_train, y_train = create_sliding_windows(train_scaled, n_his=n_his, n_pred=1)
    X_val, y_val = create_sliding_windows(val_scaled, n_his=n_his, n_pred=1)
    X_test, y_test = create_sliding_windows(test_scaled, n_his=n_his, n_pred=1)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    model = SimpleSTGCN(num_nodes=N, n_his=n_his,
                        gcn_hidden=gcn_hidden, temporal_hidden=temporal_hidden,
                        A_norm=A_norm).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    if out_dir is None:
        out_dir = PROJECT_ROOT / "outputs_stgcn" / "top50"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_file = out_dir / model_name

    best_val = float("inf")
    best_epoch = -1
    no_improve = 0

    tag = "STGCN-" + ("DELTA" if MODE.upper() == "DELTA" else "LEVEL")

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, nb = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            tr_loss += loss.item()
            nb += 1
        tr_loss /= max(nb, 1)

        model.eval()
        va_loss, nb = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_loss += loss.item()
                nb += 1
        va_loss /= max(nb, 1)

        print(f"[{tag}] Epoch {epoch:03d} | train {tr_loss:.6f} | val {va_loss:.6f}")

        if va_loss < best_val - min_delta:
            best_val = va_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), best_model_file)
            print(f"[{tag} save-best] epoch {epoch} -> {best_model_file}")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[{tag} early_stop] best_epoch={best_epoch}, best_val={best_val:.6f}")
            break

    model.load_state_dict(torch.load(best_model_file, map_location=device))
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_t).detach().cpu().numpy()

    return {
        "model": model,
        "scaler": scaler,
        "train_scaled": train_scaled,
        "test_scaled": test_scaled,
        "y_test_scaled": y_test,
        "y_pred_scaled": y_pred_test,
        "out_dir": out_dir
    }



def backtest_cds_cash_pnl_from_delta_protection(
    y_true_delta_bp: np.ndarray,   # realized Δspread (bp change)
    y_pred_delta_bp: np.ndarray,   # predicted Δspread (bp change)
    top_quantile=0.2,
    dv01_per_bp=100.0
):
    """
    Correct sign convention using protection exposure:
      - Long protection profits when spreads widen (ΔS > 0)
      - Short protection profits when spreads tighten (ΔS < 0)

    Positions built by predicted delta:
      - Long protection: largest predicted ΔS (most positive)
      - Short protection: most negative predicted ΔS

    PnL:
      pnl_t = dv01_per_bp * sum_i w_{t,i} * ΔS_{t,i}
      where w > 0 => long protection, w < 0 => short protection.
    """
    T, N = y_true_delta_bp.shape
    K = max(1, int(round(N * top_quantile)))
    pnl = np.zeros(T, dtype=np.float64)

    for t in range(T):
        scores = y_pred_delta_bp[t]
        order = np.argsort(scores)     # ascending: most negative ... most positive
        short_idx = order[:K]          # most negative predicted -> short protection
        long_idx = order[-K:]          # most positive predicted -> long protection

        w = np.zeros(N, dtype=np.float64)
        w[long_idx] = +1.0 / K
        w[short_idx] = -1.0 / K

        pnl[t] = dv01_per_bp * np.sum(w * y_true_delta_bp[t])

    mean_ret = float(pnl.mean())
    std_ret = float(pnl.std(ddof=1)) if T > 1 else 0.0
    hit = float((pnl > 0).mean())
    sharpe = np.sqrt(52) * mean_ret / std_ret if std_ret > 0 else np.nan

    return {
        "pnl": pnl,
        "mean": mean_ret,
        "std": std_ret,
        "hit": hit,
        "sharpe": sharpe
    }


def level_to_delta_from_preds(level_true, level_pred, level_prev):
    """
    Convert LEVEL predictions to DELTA:
      delta_true = level_true - level_prev
      delta_pred = level_pred - level_prev
    Shapes: (B,N)
    """
    return (level_true - level_prev), (level_pred - level_prev)


def build_prev_levels_for_test(level_full, mode_panel_T, n_his, n_train, n_val):
    """
    We need prev level (t-1) aligned with test windows.
    When MODE==LEVEL: panel_T == T_level
    When MODE==DELTA: panel_T == T_level-1
    We'll use level_full to fetch levels.

    For test windows, targets correspond to times:
      global_t = test_start + n_his + k   in panel index space
    For LEVEL mode, that's same as level index.
    For DELTA mode, panel index is (level_t - level_{t-1}) so:
      delta index t corresponds to level time t+1 (if diff computed as level[t]-level[t-1]).
      BUT we don't need prev level for DELTA cash PnL, since y_true_delta already is delta.

    This function is mainly for LEVEL -> delta conversion.
    """
    test_start = n_train + n_val
    # in LEVEL panel, y_test has length: n_test - n_his
    # targets correspond to level index: test_start + n_his ... end-1
    # prev level is one step earlier
    # return prev levels for each test target row
    # Determine number of test targets from split:
    n_test = mode_panel_T - n_train - n_val
    B_test = n_test - n_his
    prev = np.zeros((B_test, level_full.shape[1]), dtype=np.float32)
    for k in range(B_test):
        t_global = test_start + n_his + k
        prev[k] = level_full[t_global - 1]
    return prev



if __name__ == "__main__":

    panel, colnames, level_full = load_panel(mode=MODE)
    A_norm = load_adjacency()

    # align N with adjacency
    N_panel = panel.shape[1]
    N_adj = A_norm.shape[0]
    N = min(N_panel, N_adj)
    if N_panel != N_adj:
        print(f"[align] WARNING: panel N={N_panel}, adj N={N_adj} -> using N={N}")
    panel = panel[:, :N]
    colnames = colnames[:N]
    A_norm = A_norm[:N, :N]
    level_full = level_full[:, :N]

    out_dir = PROJECT_ROOT / "outputs_stgcn" / "top50"
    out_dir.mkdir(parents=True, exist_ok=True)

    T = panel.shape[0]
    n_train, n_val, n_test = split_80_10_10(T)
    print(f"[split] T={T}, n_train={n_train}, n_val={n_val}, n_test={n_test}")

    train_raw = panel[:n_train]
    val_raw = panel[n_train:n_train + n_val]
    test_raw = panel[n_train + n_val:]

    scaler_base = StandardScaler()
    train_scaled = scaler_base.fit_transform(train_raw)
    test_scaled = scaler_base.transform(test_raw)

   
    if MODE.upper() == "LEVEL":
        y_test_naive_s, y_pred_naive_s = baseline_naive_level(train_scaled, test_scaled, n_his=N_HIS)
    else:
        y_test_naive_s, y_pred_naive_s = baseline_naive_delta(train_scaled, test_scaled, n_his=N_HIS)

    naive_m = compute_metrics_scaled_and_orig(y_test_naive_s, y_pred_naive_s, scaler_base)

    y_test_ar1_s, y_pred_ar1_s = baseline_ar1(train_scaled, test_scaled, n_his=N_HIS)
    ar1_m = compute_metrics_scaled_and_orig(y_test_ar1_s, y_pred_ar1_s, scaler_base)

 
    model_pack = train_stgcn(
        panel=panel,
        A_norm=A_norm,
        n_his=N_HIS,
        device_str=DEVICE_STR,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        gcn_hidden=GCN_HIDDEN,
        temporal_hidden=TEMPORAL_HIDDEN,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        out_dir=out_dir,
        model_name=("stgcn_best_model_delta.pt" if MODE.upper() == "DELTA" else "stgcn_best_model_level.pt")
    )
    stgcn_m = compute_metrics_scaled_and_orig(model_pack["y_test_scaled"], model_pack["y_pred_scaled"], model_pack["scaler"])

   
    def pmetrics(name, m):
        print(f"\n[{name}] Out-of-sample (Scaled)  : {m['mse_s']:.6f} {m['rmse_s']:.6f} {m['r2_s']:.6f}")
        print(f"[{name}] Out-of-sample (Orig)    : {m['mse_o']:.6f} {m['rmse_o']:.6f} {m['r2_o']:.6f}")

    pmetrics("NAIVE", naive_m)
    pmetrics("AR(1)", ar1_m)
    pmetrics("STGCN", stgcn_m)

    
    prefix_base = "level" if MODE.upper() == "LEVEL" else "delta"
    save_preds_csv(naive_m["y_true_orig"], naive_m["y_pred_orig"], colnames, out_dir, f"naive_{prefix_base}")
    save_preds_csv(ar1_m["y_true_orig"], ar1_m["y_pred_orig"], colnames, out_dir, f"ar1_{prefix_base}")
    save_preds_csv(stgcn_m["y_true_orig"], stgcn_m["y_pred_orig"], colnames, out_dir, f"stgcn_{prefix_base}")

  
    if MODE.upper() == "DELTA":
        # y_true_orig already = delta bp, y_pred_orig already = delta bp
        bt_naive = backtest_cds_cash_pnl_from_delta_protection(naive_m["y_true_orig"], naive_m["y_pred_orig"],
                                                               top_quantile=TOP_QUANTILE, dv01_per_bp=DV01_PER_BP)
        bt_ar1 = backtest_cds_cash_pnl_from_delta_protection(ar1_m["y_true_orig"], ar1_m["y_pred_orig"],
                                                             top_quantile=TOP_QUANTILE, dv01_per_bp=DV01_PER_BP)
        bt_stgcn = backtest_cds_cash_pnl_from_delta_protection(stgcn_m["y_true_orig"], stgcn_m["y_pred_orig"],
                                                               top_quantile=TOP_QUANTILE, dv01_per_bp=DV01_PER_BP)
    else:
        # Convert LEVEL predictions to DELTA using previous level (t-1)
        prev_levels = build_prev_levels_for_test(
            level_full=level_full,
            mode_panel_T=T,
            n_his=N_HIS,
            n_train=n_train,
            n_val=n_val
        )
        # Align shapes to B_test
        # y_true_orig shape is (B_test, N)
        B_test = naive_m["y_true_orig"].shape[0]
        prev_levels = prev_levels[:B_test, :]

        naive_true_d, naive_pred_d = level_to_delta_from_preds(naive_m["y_true_orig"], naive_m["y_pred_orig"], prev_levels)
        ar1_true_d, ar1_pred_d = level_to_delta_from_preds(ar1_m["y_true_orig"], ar1_m["y_pred_orig"], prev_levels)
        stg_true_d, stg_pred_d = level_to_delta_from_preds(stgcn_m["y_true_orig"], stgcn_m["y_pred_orig"], prev_levels)

        bt_naive = backtest_cds_cash_pnl_from_delta_protection(naive_true_d, naive_pred_d,
                                                               top_quantile=TOP_QUANTILE, dv01_per_bp=DV01_PER_BP)
        bt_ar1 = backtest_cds_cash_pnl_from_delta_protection(ar1_true_d, ar1_pred_d,
                                                             top_quantile=TOP_QUANTILE, dv01_per_bp=DV01_PER_BP)
        bt_stgcn = backtest_cds_cash_pnl_from_delta_protection(stg_true_d, stg_pred_d,
                                                               top_quantile=TOP_QUANTILE, dv01_per_bp=DV01_PER_BP)

    print(f"\n[bt-NAIVE] mean={bt_naive['mean']:.6f} std={bt_naive['std']:.6f} hit={bt_naive['hit']:.3f} Sharpe={bt_naive['sharpe']:.3f}")
    print(f"[bt-AR1]   mean={bt_ar1['mean']:.6f} std={bt_ar1['std']:.6f} hit={bt_ar1['hit']:.3f} Sharpe={bt_ar1['sharpe']:.3f}")
    print(f"[bt-STGCN] mean={bt_stgcn['mean']:.6f} std={bt_stgcn['std']:.6f} hit={bt_stgcn['hit']:.3f} Sharpe={bt_stgcn['sharpe']:.3f}")

    # Save pnl series
    pd.DataFrame({"pnl": bt_naive["pnl"]}).to_csv(out_dir / f"bt_naive_{prefix_base}_pnl.csv", index=False)
    pd.DataFrame({"pnl": bt_ar1["pnl"]}).to_csv(out_dir / f"bt_ar1_{prefix_base}_pnl.csv", index=False)
    pd.DataFrame({"pnl": bt_stgcn["pnl"]}).to_csv(out_dir / f"bt_stgcn_{prefix_base}_pnl.csv", index=False)

    
    metrics_all = pd.DataFrame([
        {
            "Model": "NAIVE",
            "MSE_scaled": naive_m["mse_s"], "RMSE_scaled": naive_m["rmse_s"], "R2_scaled": naive_m["r2_s"],
            "MSE_orig": naive_m["mse_o"], "RMSE_orig": naive_m["rmse_o"], "R2_orig": naive_m["r2_o"],
            "BT_mean": bt_naive["mean"], "BT_std": bt_naive["std"], "BT_hit": bt_naive["hit"], "BT_sharpe": bt_naive["sharpe"],
        },
        {
            "Model": "AR(1)",
            "MSE_scaled": ar1_m["mse_s"], "RMSE_scaled": ar1_m["rmse_s"], "R2_scaled": ar1_m["r2_s"],
            "MSE_orig": ar1_m["mse_o"], "RMSE_orig": ar1_m["rmse_o"], "R2_orig": ar1_m["r2_o"],
            "BT_mean": bt_ar1["mean"], "BT_std": bt_ar1["std"], "BT_hit": bt_ar1["hit"], "BT_sharpe": bt_ar1["sharpe"],
        },
        {
            "Model": "STGCN",
            "MSE_scaled": stgcn_m["mse_s"], "RMSE_scaled": stgcn_m["rmse_s"], "R2_scaled": stgcn_m["r2_s"],
            "MSE_orig": stgcn_m["mse_o"], "RMSE_orig": stgcn_m["rmse_o"], "R2_orig": stgcn_m["r2_o"],
            "BT_mean": bt_stgcn["mean"], "BT_std": bt_stgcn["std"], "BT_hit": bt_stgcn["hit"], "BT_sharpe": bt_stgcn["sharpe"],
        },
    ])

    out_metrics = out_dir / f"metrics_{prefix_base}_all_models_with_backtest.csv"
    metrics_all.to_csv(out_metrics, index=False)

    print(f"\n[save] Saved combined metrics: {out_metrics}")
    print("\n[LaTeX] Combined Table:")
    print(metrics_all.to_latex(index=False, float_format="%.4f"))

    
    try:
        import vectorbt as vbt

        pnl = bt_stgcn["pnl"]
        equity = INIT_CASH + np.cumsum(pnl)
        close = pd.Series(equity, name="LS_EQUITY")

        vbt.settings.array_wrapper['freq'] = 'W'
        vbt.settings.returns['year_freq'] = '365D'

        entries = pd.Series(True, index=close.index)
        exits = pd.Series(False, index=close.index)

        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=INIT_CASH,
            fees=FEE,
            freq="W"
        )

        stats = pf.stats()
        stats.to_csv(out_dir / f"vectorbt_{prefix_base}_stgcn_stats.csv")
        pd.DataFrame({"equity": close.values}).to_csv(out_dir / f"vectorbt_{prefix_base}_stgcn_equity.csv", index=False)

        print("\n[vectorbt] STGCN equity stats:")
        print(stats)

    except ImportError:
        print("\n[vectorbt] not installed -> skipped.")
