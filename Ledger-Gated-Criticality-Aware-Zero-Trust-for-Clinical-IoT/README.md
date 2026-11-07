# Ledger-Gated, Criticality-Aware Zero Trust for Clinical IoT:Evaluation on TON_IoT Flows with a Simulated Identity Layer

Run `BZT_SHIS_v2.ipynb` top to bottom (about 50 seconds). Needs numpy, pandas, scikit-learn, matplotlib, cryptography.
Keep these next to the notebook: `toniot_zt_sessions.csv`, `device_registry_toniot.csv`, `build_toniot_sessions.py`, `generate_dataset.py`.
If the two prepared CSVs are missing, put `Train_Test_Network.csv` next to the notebook and it rebuilds them.

## What is real and what is added
| Part | Status |
|---|---|
| Network features, normal flows, 9 attack classes (backdoor, ddos, dos, injection, mitm, password, ransomware, scanning, xss) | **Real TON_IoT** (`Train_Test_Network.csv`) |
| 20,569 exact duplicate rows | Removed before sampling (they would sit in both train and test) |
| Device DIDs, registry, revocations, signatures, nonces | Added (simulated identity layer) |
| Spoofing, replay, unregistered DID, revoked-device use | **Injected** at known rates (40% of attacks); their traffic is a real normal flow. Column `source` tells you which rows these are |
| Hospital | Derived from the victim (destination) IP, so sites differ (non-IID) |
| Device type and criticality | Assigned at random; they drive the policy layer, not the classifier signal |
| Timestamps | **Synthetic** (the file has none), independent of the label |
| Raw `src_ip` | Never a feature (it separates attacker from normal machines). Kept as `orig_src_ip` for the hold-out check |

Rebuild with another seed or attack mix: `python build_toniot_sessions.py --csv Train_Test_Network.csv --n 40000 --identity_share 0.4`.
For a real timeline, use the `Processed_datasets` network files (they have a `ts` column); the builder uses it automatically.

## Results (test = latest 25% of 39,997 sessions, 21% attacks)
| Method | Attacks caught | Legit blocked | Expected cost/session |
|---|---|---|---|
| RF / HGB / MLP, network features | 55-57% | 0.3-0.7% | 1.28-1.33 |
| Ledger gate alone | 40.9% | 0.3% | 1.76 |
| Gate + RF | 97.3% | 0.9% | 0.12 |
| + trust history (A4) | 97.0% | 0.5% | 0.11 |
| + criticality tiers (A5) | 97.7% (restrict counted) | 1.3% (0.5% hard denied) | 0.10 |

Three forward-in-time folds: A4 F1 0.974 ± 0.003 vs ML-only 0.725 ± 0.018.

## What the results actually say
- **On real network attacks the ML-only model is already excellent**: 8 of 9 classes are caught at 99-100% (mitm 91.7%). TON_IoT is known to be easy to separate.
  Holding out 30% of source machines still gives ROC-AUC 0.9997 (recall 93.5% vs 98.7% on a random split), so it is not only an IP shortcut, but expect optimistic numbers.
- **Every point of the ML-only vs full gap comes from the injected identity attacks** (ML-only catches about 1% of them, the gate+model about 90-100%). That gap is a property of how they were injected, not a finding about real hospitals.
- **Trust history adds little** (F1 +0.005 over the stacked baseline A3b) and has no real timeline behind it here.
- **Criticality tiers barely help on this data** (cost -12% vs a fixed 0.5 threshold; life-critical denials unchanged at 0.63%) because criticality is random and the model is near-perfect. Their value needs real clinical costs and a real device mapping.
- **Evasion (fixed, with a caveat).** The attacker model throttles a flow to the size of a median normal flow (all byte/packet counts scaled by one factor, rounded to integers; protocol, service and state untouched). With standard training, detection of network attacks falls from 98.5% to **67.4%** for the full framework (ML-only RF: 85.5%) - the full pipeline degraded *more* because stage 2 had only ever seen confident stage-1 scores. With **evasion-aware training** (perturbed copies of training attacks added to both stages, stage-1 score recomputed out-of-fold) it stays at **99%** across all strengths.
  - Caveat 1: training and test use the same perturbation family, so this shows robustness to *throttling*, not to an adaptive attacker.
  - Caveat 2: a control applies the same transform to *legitimate* flows. The evasion-aware full model flags 7.4% of them at strength 0.75 and 32.5% at full mimicry (the ML-only evasion-aware RF: 21% and 36.5%), so part of the gain at the extreme end reflects the transform's fingerprint. Real legitimate traffic is not perturbed, and its false-reject rate only rises from 0.5% to 0.7%.
  - An earlier version of this fix (independent per-feature shrinking, non-integer counts) was discarded because the model learned the fingerprint and flagged 95% of un-perturbed identity attacks; the control test exists to catch exactly that.
- **Federated ensemble (fixed).** Raw federated probabilities were miscalibrated for one site (hospital 3 F1 0.446). Per-site isotonic calibration on each hospital's own calibration window (only the 1-D map is used, no data is shared) lifts it to 0.915 and the mean F1 from 0.866 to **0.962** (local-only 0.971, centralised 0.966). Mean AUC is still slightly lower than centralised (0.982 vs 0.990), and for hospital 3 the federated ensemble is worse than its local model (AUC 0.957 vs 0.988).
- Ledger: 6/6 tamper and rogue-write attempts caught; a colluding quorum of 3 of 4 validators is accepted by design. Message overhead is 44 bytes on a ~63-byte payload.

## Before you cite numbers
Say "real TON_IoT flows with an injected identity layer". Do not describe the identity results as measured on real attacks. A stronger version needs real device identities or a second real dataset, a real timeline, and clinician-derived costs.
