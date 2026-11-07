"""
HealthIoT-ZT-Synth  -  synthetic "novelty-level" dataset generator for BZT-SHIS v2
==================================================================================

WHY THIS EXISTS
  The original notebook used TON_IoT network flows only.  Those flows carry no
  identity, ledger or clinical context, so the novel parts of the framework
  (ledger-gated access, on-chain trust history, criticality-aware decisions,
  per-hospital federated training) cannot be tested on them.

  This generator produces two linked tables:
    * device_registry.csv      - what the permissioned ledger is seeded with
    * healthiot_zt_sessions.csv - one row per access request (session)

IMPORTANT - READ BEFORE CITING RESULTS
  The data is SYNTHETIC.  Identity attacks (spoofing, replay, revoked-device use,
  unregistered DIDs) are generated with network behaviour that looks like normal
  traffic, because that is what makes them hard for a behaviour-only model.
  Therefore the *size* of the gap between ML-only and the full framework is a
  property of this generator, not evidence about the real world.  Use it to
  develop and test the mechanism, then validate on real data (e.g. TON_IoT /
  Edge-IIoTset flows merged with the same identity fields) before claiming
  numbers in a paper.  Every knob that controls difficulty is a parameter below.

Usage:
    python generate_dataset.py                       # writes CSVs next to the script
    python generate_dataset.py --n 50000 --seed 7
"""
import argparse
import hashlib
import os

import numpy as np
import pandas as pd

# median profile per device type:
#   (criticality, data_sensitivity, duration_s, src_bytes, dst_bytes, pkts, req_per_min, rtt_ms)
# criticality 4 = life-critical ... 1 = low ; data_sensitivity 4 = most sensitive
DEVICE_TYPES = {
    "ventilator":      (4, 3, 600,  4_000,     1_500,   60,   12, 10),
    "infusion_pump":   (4, 3, 300,  2_500,       900,   40,    6, 12),
    "cardiac_monitor": (3, 4, 900, 12_000,     2_000,  150,   30, 15),
    "glucose_monitor": (3, 4, 120,    800,       300,   12,    3, 25),
    "pulse_oximeter":  (2, 3, 240,  1_500,       500,   25,   10, 20),
    "smart_wearable":  (2, 3,  90,  1_200,       400,   20,    4, 40),
    "imaging_gateway": (2, 4, 400, 4_000_000, 200_000, 3000,    2,  8),
    "env_sensor":      (1, 1,  60,    300,       120,    6,    1, 30),
}
TYPE_NAMES = list(DEVICE_TYPES)

ATTACKS = {  # attack_type: (share of attacks, family)
    "DoS Flood":            (0.17, "network"),
    "Data Exfiltration":    (0.10, "network"),
    "Data Injection":       (0.15, "data"),
    "Man-in-the-Middle":    (0.10, "network"),
    "Device Spoofing":      (0.18, "identity"),
    "Replay Attack":        (0.13, "identity"),
    "Unregistered Device":  (0.10, "identity"),
    "Revoked Device Use":   (0.07, "identity"),
}

DAY = 86_400


def _h(s, n=32):
    return hashlib.sha256(s.encode()).hexdigest()[:n]


def generate(n_sessions=30_000, n_devices=600, n_hospitals=5, days=30,
             attack_rate=0.20, revoked_share=0.04, net_shift=1.0, seed=42):
    """net_shift: 1.0 = full network signature for network attacks, 0.5 = half
    as strong (harder), 0.0 = attacks look like normal traffic."""
    rng = np.random.default_rng(seed)
    t0 = 1_767_225_600  # 2026-01-01 00:00:00 UTC
    horizon = days * DAY

    # ------------------------------------------------------------------ registry
    hosp_type_p = rng.dirichlet(np.ones(len(TYPE_NAMES)) * 1.5, size=n_hospitals)
    dev_h = rng.integers(0, n_hospitals, n_devices)
    dev_t = np.array([rng.choice(len(TYPE_NAMES), p=hosp_type_p[h]) for h in dev_h])
    fw_ver = rng.integers(1, 4, n_devices)
    reg = pd.DataFrame({
        "did": [_h(f"dev{i}|{TYPE_NAMES[dev_t[i]]}|{fw_ver[i]}|{seed}") for i in range(n_devices)],
        "device_type": [TYPE_NAMES[k] for k in dev_t],
        "criticality": [DEVICE_TYPES[TYPE_NAMES[k]][0] for k in dev_t],
        "data_sensitivity": [DEVICE_TYPES[TYPE_NAMES[k]][1] for k in dev_t],
        "hospital_id": dev_h,
        "fw_hash": [_h(f"{TYPE_NAMES[dev_t[i]]}-fw{fw_ver[i]}", 16) for i in range(n_devices)],
        "registered_at": t0,
    })
    revoked = rng.random(n_devices) < revoked_share
    reg["revoked_at"] = np.where(revoked, t0 + rng.integers(8 * DAY, 25 * DAY, n_devices), np.nan)
    activity = rng.lognormal(0, 0.6, n_devices)
    activity /= activity.sum()

    prof = np.array([DEVICE_TYPES[k][2:] for k in TYPE_NAMES], dtype=float)

    def net_features(types_idx, noise=1.0):
        p = prof[types_idx]
        n = len(types_idx)
        ln = lambda col, s: p[:, col] * rng.lognormal(0, s * noise, n)
        dur, sb, db, pk, rpm, rtt = ln(0, .35), ln(1, .40), ln(2, .40), ln(3, .35), ln(4, .30), ln(5, .25)
        return dict(duration_s=dur, src_bytes=sb, dst_bytes=db, src_pkts=pk, req_per_min=rpm,
                    rtt_ms=rtt, jitter_ms=rtt * rng.lognormal(np.log(.1), .4, n),
                    payload_entropy=np.clip(rng.normal(4.2, .3, n), 0, 8),
                    vitals_zscore=rng.normal(0, .8, n))

    def nonces(n):
        return [f"{x:016x}" for x in rng.integers(0, 2**62, n)]

    def frame(feats, **cols):
        d = pd.DataFrame(feats)
        for k, v in cols.items():
            d[k] = v
        return d

    parts = []
    n_att = int(n_sessions * attack_rate)
    n_norm = n_sessions - n_att

    # ------------------------------------------------------------------ normal traffic
    di = rng.choice(n_devices, size=n_norm * 2, p=activity)
    ts = t0 + rng.random(n_norm * 2) * horizon
    ra = reg["revoked_at"].values[di]
    ok = np.isnan(ra) | (ts < ra)
    di, ts = di[ok][:n_norm], ts[ok][:n_norm]
    nn = len(di)
    f = net_features(dev_t[di])
    burst = rng.random(nn) < .03            # benign bursts (bulk upload, resync)
    f["src_bytes"] = np.where(burst, f["src_bytes"] * rng.uniform(5, 15, nn), f["src_bytes"])
    f["duration_s"] = np.where(burst, f["duration_s"] * rng.uniform(2, 4, nn), f["duration_s"])
    det = rng.random(nn) < .02              # genuine patient deterioration (clinically real)
    f["vitals_zscore"] = np.where(det, rng.normal(0, 1, nn) + rng.choice([-3.5, 3.5], nn), f["vitals_zscore"])
    spike = rng.random(nn) < .015
    f["rtt_ms"] = np.where(spike, f["rtt_ms"] * rng.uniform(2.5, 4, nn), f["rtt_ms"])
    fw = reg["fw_hash"].values[di].copy()
    stale = rng.random(nn) < .015           # legit firmware update not yet re-registered
    fw[stale] = [_h(f"upd{i}", 16) for i in np.where(stale)[0]]
    sig = (rng.random(nn) > .003).astype(int)  # clock / transport glitch
    base = frame(f, timestamp=ts, hospital_id=reg["hospital_id"].values[di], claimed_did=reg["did"].values[di],
                 device_type=reg["device_type"].values[di], presented_fw_hash=fw, sig_ok=sig,
                 nonce=nonces(nn), label=0, attack_type="Normal", attack_family="none")
    parts.append(base)

    # per-hospital attack mix (non-IID across hospitals)
    hosp_w = {a: rng.dirichlet(np.ones(n_hospitals) * 2.0) for a in ATTACKS}
    counts = {a: int(round(n_att * s)) for a, (s, _) in ATTACKS.items()}
    unrevoked = np.where(~revoked)[0]

    def attack_rows(name, k, feats, did, dtype, hosp, fwh, sg, ts_, nonce=None):
        return frame(feats, timestamp=ts_, hospital_id=hosp, claimed_did=did, device_type=dtype,
                     presented_fw_hash=fwh, sig_ok=sg, nonce=nonce if nonce is not None else nonces(k),
                     label=1, attack_type=name, attack_family=ATTACKS[name][1])

    s = net_shift

    def pick_victims(k):  # compromised / victim registered device
        i = rng.choice(unrevoked, k)
        return i, t0 + rng.random(k) * horizon

    # DoS
    k = counts["DoS Flood"]; i, tt = pick_victims(k); f = net_features(dev_t[i])
    slow = rng.random(k) < .15              # low-and-slow variant
    mult = np.where(slow, rng.uniform(2, 4, k), rng.uniform(25, 80, k)) ** s
    f["req_per_min"] *= mult; f["src_pkts"] *= mult ** .8
    f["duration_s"] *= np.where(slow, 1, rng.uniform(.2, 1, k))
    parts.append(attack_rows("DoS Flood", k, f, reg["did"].values[i], reg["device_type"].values[i],
                             reg["hospital_id"].values[i], reg["fw_hash"].values[i], np.ones(k, int), tt))
    # Exfiltration
    k = counts["Data Exfiltration"]; i, tt = pick_victims(k); f = net_features(dev_t[i])
    low = rng.random(k) < .2
    m = np.where(low, rng.uniform(2, 4, k), rng.uniform(8, 40, k)) ** s
    f["src_bytes"] *= m; f["duration_s"] *= rng.uniform(2, 5, k) ** s
    f["payload_entropy"] = np.clip(rng.normal(4.2 + 2.8 * s, .4, k), 0, 8)
    tt = (tt // DAY) * DAY + rng.uniform(0, 5 * 3600, k)       # off-hours
    parts.append(attack_rows("Data Exfiltration", k, f, reg["did"].values[i], reg["device_type"].values[i],
                             reg["hospital_id"].values[i], reg["fw_hash"].values[i], np.ones(k, int), tt))
    # Data injection (valid identity, implausible vitals)
    k = counts["Data Injection"]; i, tt = pick_victims(k); f = net_features(dev_t[i])
    strong = rng.random(k) < .75
    f["vitals_zscore"] = np.where(strong, rng.normal(0, 1.2, k) + rng.choice([-4, 4], k), rng.normal(0, 1.2, k))
    f["payload_entropy"] = np.clip(rng.normal(4.4, .3, k), 0, 8)
    parts.append(attack_rows("Data Injection", k, f, reg["did"].values[i], reg["device_type"].values[i],
                             reg["hospital_id"].values[i], reg["fw_hash"].values[i], np.ones(k, int), tt))
    # MITM
    k = counts["Man-in-the-Middle"]; i, tt = pick_victims(k); f = net_features(dev_t[i])
    hit = rng.random(k) < .7
    f["rtt_ms"] = np.where(hit, f["rtt_ms"] * rng.uniform(2.5, 5, k) ** s, f["rtt_ms"])
    f["jitter_ms"] = np.where(hit, f["jitter_ms"] * 3, f["jitter_ms"])
    sg = (rng.random(k) > .6).astype(int)
    parts.append(attack_rows("Man-in-the-Middle", k, f, reg["did"].values[i], reg["device_type"].values[i],
                             reg["hospital_id"].values[i], reg["fw_hash"].values[i], sg, tt))
    # Spoofing: valid victim DID, normal-looking traffic, attacker lacks the key (8% stolen keys)
    k = counts["Device Spoofing"]; i, tt = pick_victims(k); f = net_features(dev_t[i], noise=1.15)
    sg = (rng.random(k) < .08).astype(int)
    wrong_fw = rng.random(k) < .5
    fwh = np.where(wrong_fw, [_h(f"guess{j}", 16) for j in range(k)], reg["fw_hash"].values[i])
    parts.append(attack_rows("Device Spoofing", k, f, reg["did"].values[i], reg["device_type"].values[i],
                             reg["hospital_id"].values[i], fwh, sg, tt))
    # Replay: re-send an earlier legit nonce of the same DID
    k = counts["Replay Attack"]
    src = base.iloc[rng.choice(len(base), k)].reset_index(drop=True)
    f = net_features(np.array([TYPE_NAMES.index(x) for x in src["device_type"]]), noise=1.1)
    fresh = rng.random(k) < .10               # 10% attackers also forge a fresh nonce
    nonce = np.where(fresh, nonces(k), src["nonce"].values)
    parts.append(attack_rows("Replay Attack", k, f, src["claimed_did"].values, src["device_type"].values,
                             src["hospital_id"].values, src["presented_fw_hash"].values, np.ones(k, int),
                             src["timestamp"].values + rng.uniform(60, 7200, k), nonce))
    # Unregistered DID
    k = counts["Unregistered Device"]
    tidx = rng.integers(0, len(TYPE_NAMES), k); f = net_features(tidx, noise=1.15)
    hosp = rng.choice(n_hospitals, k, p=hosp_w["Unregistered Device"])
    parts.append(attack_rows("Unregistered Device", k, f, [_h(f"rogue{j}|{seed}") for j in range(k)],
                             [TYPE_NAMES[j] for j in tidx], hosp, [_h(f"rfw{j}", 16) for j in range(k)],
                             (rng.random(k) < .5).astype(int), t0 + rng.random(k) * horizon))
    # Revoked device continues to transmit after revocation
    k = counts["Revoked Device Use"]
    rv = np.where(revoked)[0]
    i = rng.choice(rv, k); ra = reg["revoked_at"].values[i]
    tt = ra + rng.random(k) * (t0 + horizon - ra); f = net_features(dev_t[i])
    parts.append(attack_rows("Revoked Device Use", k, f, reg["did"].values[i], reg["device_type"].values[i],
                             reg["hospital_id"].values[i], reg["fw_hash"].values[i], np.ones(k, int), tt))

    df = pd.concat(parts, ignore_index=True)
    df = df.merge(reg[["device_type", "criticality", "data_sensitivity"]].drop_duplicates("device_type"),
                  on="device_type", how="left")
    df["data_volume_kb"] = (df["src_bytes"] + df["dst_bytes"]) / 1024.0
    df["hour_of_day"] = ((df["timestamp"] % DAY) // 3600).astype(int)
    df["timestamp"] = df["timestamp"].round(0).astype(np.int64)
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df.insert(0, "session_id", [f"S{i:06d}" for i in range(len(df))])
    for c in ["duration_s", "src_bytes", "dst_bytes", "src_pkts", "req_per_min", "rtt_ms", "jitter_ms",
              "payload_entropy", "vitals_zscore", "data_volume_kb"]:
        df[c] = df[c].round(3)
    cols = ["session_id", "timestamp", "hour_of_day", "hospital_id", "claimed_did", "device_type",
            "criticality", "data_sensitivity", "presented_fw_hash", "sig_ok", "nonce",
            "duration_s", "src_bytes", "dst_bytes", "src_pkts", "req_per_min", "data_volume_kb",
            "rtt_ms", "jitter_ms", "payload_entropy", "vitals_zscore",
            "label", "attack_type", "attack_family"]
    return df[cols], reg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30_000)
    ap.add_argument("--devices", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--net_shift", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    d, r = generate(a.n, a.devices, net_shift=a.net_shift, seed=a.seed)
    d.to_csv(os.path.join(a.out, "healthiot_zt_sessions.csv"), index=False)
    r.to_csv(os.path.join(a.out, "device_registry.csv"), index=False)
    print(f"sessions={len(d):,} attack_rate={d.label.mean():.3f} devices={len(r)}")
    print(d.attack_type.value_counts().to_string())
