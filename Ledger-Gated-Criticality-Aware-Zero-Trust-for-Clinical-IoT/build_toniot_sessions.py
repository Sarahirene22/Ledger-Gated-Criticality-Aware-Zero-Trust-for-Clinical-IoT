"""
Turns the REAL TON_IoT network flows (Train_Test_Network.csv) into ledger-ready access sessions.

What is real:      every network feature (duration, bytes, packets, ports, protocol, service, state, ...)
                   and every network attack (backdoor, ddos, dos, injection, mitm, password, ransomware, scanning, xss).
What is added:     a device identity layer that TON_IoT does not have - DIDs, a registry with revocations, signatures,
                   nonces, hospitals, device criticality - and four identity attacks (spoofing, replay, unregistered
                   DID, revoked-device use) injected at known rates.  Injected attacks reuse REAL normal flows for their
                   network behaviour (that is what a spoofer / replayer would send), and are labelled source=injected_identity.

Things you must know (also in the README):
  * Train_Test_Network.csv has NO timestamp column.  Timestamps here are synthetic (random, independent of the label) unless
    the input has a 'ts' column (the Processed_datasets Network_dataset_*.csv files do - use those for a real timeline).
  * Hospital/site is derived from the destination (victim) IP.  Raw src_ip is NOT used as a device id: in TON_IoT it almost perfectly separates attacker machines from normal machines,
    which would leak the label.  It is kept as `orig_src_ip` for grouping/leakage checks only.
  * Exact duplicate rows (about 9.7 % of the file) are removed before sampling.
  * Device roles / criticality are assigned to registry entries at random; they drive the POLICY layer, not the classifier signal.

Usage:  python build_toniot_sessions.py --csv Train_Test_Network.csv --n 40000
"""
import argparse, hashlib, os
import numpy as np
import pandas as pd

DAY = 86_400
T0 = 1_767_225_600
DEVICE_TYPES = {  # name: (criticality, data_sensitivity)
    "ventilator": (4, 3), "infusion_pump": (4, 3), "cardiac_monitor": (3, 4), "glucose_monitor": (3, 4),
    "pulse_oximeter": (2, 3), "smart_wearable": (2, 3), "imaging_gateway": (2, 4), "env_sensor": (1, 1)}
TYPE_NAMES = list(DEVICE_TYPES)
IDENT_MIX = {"Device Spoofing": .375, "Replay Attack": .27, "Unregistered Device": .21, "Revoked Device Use": .145}

NUM = ["duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "src_ip_bytes", "dst_ip_bytes", "missed_bytes",
       "dst_port", "http_request_body_len", "http_response_body_len", "http_status_code", "dns_qtype", "dns_rcode"]


def _h(s, n=32):
    return hashlib.sha256(s.encode()).hexdigest()[:n]


def build(csv, n_sessions=40_000, attack_rate=.20, identity_share=.40, n_devices=600, n_hospitals=5,
          days=30, revoked_share=.04, seed=42):
    rng = np.random.default_rng(seed)
    horizon = days * DAY
    raw = pd.read_csv(csv, low_memory=False)
    n_raw = len(raw)
    d = raw.drop_duplicates().reset_index(drop=True)
    print(f"raw rows {n_raw:,} -> {len(d):,} after removing {n_raw - len(d):,} exact duplicates")

    # ---- real feature engineering -------------------------------------------------------------
    cat = {c: {v: i for i, v in enumerate(sorted(d[c].astype(str).unique()))} for c in ["proto", "service", "conn_state"]}
    f = pd.DataFrame({"duration_s": d["duration"].astype(float)})
    for c in ["src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "src_ip_bytes", "dst_ip_bytes", "missed_bytes", "dst_port",
              "http_request_body_len", "http_response_body_len", "http_status_code", "dns_qtype", "dns_rcode"]:
        f[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    f["data_volume_kb"] = (f["src_bytes"] + f["dst_bytes"]) / 1024.0
    for c in ["proto", "service", "conn_state"]:
        f[c + "_code"] = d[c].astype(str).map(cat[c]).astype(int)
    f["ssl_established_flag"] = (d["ssl_established"].astype(str) == "T").astype(int)
    f["weird_flag"] = (d["weird_name"].astype(str) != "-").astype(int)
    f["dns_flag"] = (d["dns_query"].astype(str) != "-").astype(int)
    FEATS = list(f.columns)
    f["orig_src_ip"] = d["src_ip"].values
    f["type"] = d["type"].values
    f["hosp_h"] = [int(hashlib.md5(ip.encode()).hexdigest(), 16) % n_hospitals for ip in d["dst_ip"].astype(str)]   # site = victim network, so attackers hit several sites
    if "ts" in d.columns:
        t = pd.to_numeric(d["ts"], errors="coerce").values
        f["tt"] = T0 + (t - np.nanmin(t)) / max(np.nanmax(t) - np.nanmin(t), 1) * horizon; ts_syn = False
    else:
        f["tt"] = np.nan; ts_syn = True

    # ---- device registry ----------------------------------------------------------------------
    hosp_type_p = rng.dirichlet(np.ones(len(TYPE_NAMES)) * 1.5, size=n_hospitals)
    dev_h = np.arange(n_devices) % n_hospitals                      # every hospital owns devices
    dev_t = np.array([rng.choice(len(TYPE_NAMES), p=hosp_type_p[h]) for h in dev_h])
    fw_ver = rng.integers(1, 4, n_devices)
    reg = pd.DataFrame({
        "did": [_h(f"dev{i}|{TYPE_NAMES[dev_t[i]]}|{fw_ver[i]}|{seed}") for i in range(n_devices)],
        "device_type": [TYPE_NAMES[k] for k in dev_t],
        "criticality": [DEVICE_TYPES[TYPE_NAMES[k]][0] for k in dev_t],
        "data_sensitivity": [DEVICE_TYPES[TYPE_NAMES[k]][1] for k in dev_t],
        "hospital_id": dev_h,
        "fw_hash": [_h(f"{TYPE_NAMES[dev_t[i]]}-fw{fw_ver[i]}", 16) for i in range(n_devices)],
        "registered_at": T0})
    revoked = rng.random(n_devices) < revoked_share
    rev_at = np.where(revoked, T0 + rng.integers(8 * DAY, 25 * DAY, n_devices), np.nan)
    reg["revoked_at"] = rev_at
    act = rng.lognormal(0, .6, n_devices)

    def pick_devices(hosp, ts):
        out = np.empty(len(hosp), int)
        for h in range(n_hospitals):
            m = np.where(hosp == h)[0]
            if not len(m): continue
            pool = np.where(dev_h == h)[0]
            cand = rng.choice(pool, len(m), p=act[pool] / act[pool].sum())
            bad = ~np.isnan(rev_at[cand]) & (rev_at[cand] <= ts[m])
            safe = pool[~revoked[pool]] if (~revoked[pool]).any() else pool
            cand[bad] = rng.choice(safe, bad.sum())
            out[m] = cand
        return out

    def nonces(k):
        return [f"{x:016x}" for x in rng.integers(0, 2**62, k)]

    def take(frame, k):
        return frame.sample(k, random_state=int(rng.integers(0, 2**31))).copy()

    n_att = int(n_sessions * attack_rate); n_norm = n_sessions - n_att
    n_id = int(n_att * identity_share); n_real = n_att - n_id
    normal_pool = f[f.type == "normal"]
    norm = take(normal_pool, n_norm)
    atk_pool = f[f.type != "normal"]
    types = sorted(atk_pool.type.unique())
    per = n_real // len(types)
    real_atk = pd.concat([take(atk_pool[atk_pool.type == t], min(per, (atk_pool.type == t).sum())) for t in types])
    print(f"real attack flows per type: {real_atk.type.value_counts().to_dict()}")

    def finish(frame, source, label, attack_type, family, did, dtype, hosp, fwh, sg, ts, nonce):
        out = frame[FEATS + ["orig_src_ip"]].copy().reset_index(drop=True)
        out.insert(0, "timestamp", ts); out.insert(1, "hospital_id", hosp); out.insert(2, "claimed_did", did)
        out.insert(3, "device_type", dtype); out["presented_fw_hash"] = fwh; out["sig_ok"] = sg; out["nonce"] = nonce
        out["label"] = label; out["attack_type"] = attack_type; out["attack_family"] = family; out["source"] = source
        return out

    parts = []
    # normal real flows from legitimate registered devices
    ts = frame_ts = np.where(np.isnan(norm.tt.values), T0 + rng.random(len(norm)) * horizon, norm.tt.values)
    di = pick_devices(norm.hosp_h.values, ts)
    fw = reg.fw_hash.values[di].copy(); stale = rng.random(len(norm)) < .015
    fw[stale] = [_h(f"upd{i}", 16) for i in np.where(stale)[0]]
    normal_df = finish(norm, "ton_iot_real", 0, "Normal", "none", reg.did.values[di], reg.device_type.values[di],
                       reg.hospital_id.values[di], fw, (rng.random(len(norm)) > .003).astype(int), ts, nonces(len(norm)))
    parts.append(normal_df)
    # real network attacks sent by compromised, still-valid registered devices
    ts = np.where(np.isnan(real_atk.tt.values), T0 + rng.random(len(real_atk)) * horizon, real_atk.tt.values)
    di = pick_devices(real_atk.hosp_h.values, ts)
    parts.append(finish(real_atk, "ton_iot_real", 1, real_atk.type.values, "network", reg.did.values[di], reg.device_type.values[di],
                        reg.hospital_id.values[di], reg.fw_hash.values[di], np.ones(len(real_atk), int), ts, nonces(len(real_atk))))

    # injected identity attacks - network behaviour is a REAL normal flow
    counts = {a: int(round(n_id * s)) for a, s in IDENT_MIX.items()}
    unrev = np.where(~revoked)[0]
    def donors(k): return take(normal_pool, k)
    k = counts["Device Spoofing"]; v = rng.choice(unrev, k); dn = donors(k)
    wrong = rng.random(k) < .5
    parts.append(finish(dn, "injected_identity", 1, "Device Spoofing", "identity", reg.did.values[v], reg.device_type.values[v],
                        reg.hospital_id.values[v], np.where(wrong, [_h(f"guess{j}", 16) for j in range(k)], reg.fw_hash.values[v]),
                        (rng.random(k) < .08).astype(int), T0 + rng.random(k) * horizon, nonces(k)))
    k = counts["Replay Attack"]; src = normal_df.sample(k, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True); dn = donors(k)
    fresh = rng.random(k) < .10
    parts.append(finish(dn, "injected_identity", 1, "Replay Attack", "identity", src.claimed_did.values, src.device_type.values,
                        src.hospital_id.values, src.presented_fw_hash.values, np.ones(k, int),
                        src.timestamp.values + rng.uniform(60, 7200, k), np.where(fresh, nonces(k), src.nonce.values)))
    k = counts["Unregistered Device"]; dn = donors(k)
    parts.append(finish(dn, "injected_identity", 1, "Unregistered Device", "identity", [_h(f"rogue{j}|{seed}") for j in range(k)],
                        [TYPE_NAMES[j] for j in rng.integers(0, len(TYPE_NAMES), k)], rng.integers(0, n_hospitals, k),
                        [_h(f"rfw{j}", 16) for j in range(k)], (rng.random(k) < .5).astype(int), T0 + rng.random(k) * horizon, nonces(k)))
    k = counts["Revoked Device Use"]; rv = rng.choice(np.where(revoked)[0], k); dn = donors(k); ra = rev_at[rv]
    parts.append(finish(dn, "injected_identity", 1, "Revoked Device Use", "identity", reg.did.values[rv], reg.device_type.values[rv],
                        reg.hospital_id.values[rv], reg.fw_hash.values[rv], np.ones(k, int),
                        ra + rng.random(k) * (T0 + horizon - ra), nonces(k)))

    out = pd.concat(parts, ignore_index=True)
    out = out.merge(reg[["device_type", "criticality", "data_sensitivity"]].drop_duplicates("device_type"), on="device_type", how="left")
    out["timestamp"] = out["timestamp"].round(0).astype(np.int64)
    out["ts_is_synthetic"] = int(ts_syn)
    out = out.sort_values("timestamp", kind="stable").reset_index(drop=True)
    out.insert(0, "session_id", [f"S{i:06d}" for i in range(len(out))])
    cols = (["session_id", "timestamp", "hospital_id", "claimed_did", "device_type", "criticality", "data_sensitivity",
             "presented_fw_hash", "sig_ok", "nonce"] + FEATS +
            ["label", "attack_type", "attack_family", "source", "orig_src_ip", "ts_is_synthetic"])
    return out[cols], reg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="Train_Test_Network.csv")
    ap.add_argument("--n", type=int, default=40_000)
    ap.add_argument("--identity_share", type=float, default=.40, help="share of attacks that are injected identity attacks")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    s, r = build(a.csv, a.n, identity_share=a.identity_share, seed=a.seed)
    s.to_csv(os.path.join(a.out, "toniot_zt_sessions.csv"), index=False)
    r.to_csv(os.path.join(a.out, "device_registry_toniot.csv"), index=False)
    print(f"sessions={len(s):,} attack_rate={s.label.mean():.3f}")
    print(s.groupby(["source", "attack_type"]).size().to_string())
    print("per-hospital attack share:", s.groupby("hospital_id").label.mean().round(3).to_dict())
