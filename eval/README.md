# Evaluation

Objective tracking evaluation for the webcam-mocap pipeline. Feeds the abba
test clips (`/org/jonasjohansson/abba/assets/test-clips`) through the **same**
MediaPipe Holistic → Kalidokit → VRM path the app uses, and measures:

- **detection rates** — face / pose / world / left-hand / right-hand
- **legVis** — MediaPipe confidence on knees + ankles (0–1)
- **bone movement** — variance of each VRM bone's quaternion over the run
  (proves bones are actually being driven, not just detected)

Requires the localhost server running and `playwright` (chromium).
Output (screenshots + JSON) lands in `eval/out/` (gitignored).

## The test procedure

Single-camera mocap only works with **one clearly-framed, full-body subject**.
The suite below established which clips qualify; the procedure runs those.

```bash
python3 eval/test_procedure.py          # canonical set, pass/fail, exit code
python3 eval/run_suite.py               # sweep ALL clips, re-classify
python3 eval/evaluate.py Tai-Chi webcam # one clip, full JSON + screenshots
```

`evaluate.py` accepts a dropdown keyword (e.g. `Tai-Chi`) or an absolute web
path to any video.

## Canonical test set (single-person, clearly visible)

| Clip | pose | legVis | leg movement |
|---|---|---|---|
| **Tai-Chi** (`pexels-2882793`) | 100% | 0.85 | 0.50 |
| **Aerobics** (`pixabay-149195`) | 100% | 0.98 | 0.64 |

Pass thresholds (`test_procedure.py`): pose ≥ 90%, legVis ≥ 0.55, leg movement ≥ 0.05.

### Excluded clips (and why)

| Clip | pose | legVis | reason |
|---|---|---|---|
| Dancing ×2 | 70–80% | 0.18 | group / subjects small in frame |
| Disco ×2 | 70–80% | 0.24 | group / subjects small in frame |
| Aerobics `pixabay-75644` | 0% | 0 | subject too small/far to detect |

These are kept in the app dropdown (labelled "group, partial") for comparison,
but are **not** part of the pass/fail procedure — they fail for input reasons,
not pipeline reasons.

## What this established

The legs track correctly **when given clean single-person full-body input**
(Tai-Chi/Aerobics: 100% pose, high legVis, leg bones move). Apparent "legs
don't follow" on a live webcam is an input-framing problem (seated / partial /
multi-person), not a rigging bug. For live use: one person, full body in frame,
stand back, good light — or use Idle leg mode for the common seated case.
