# Registered predictions: VGGT-Omega against VGGT

Written BEFORE the first run of `notebooks/05d_metrics_vggt.ipynb`. Kept apart
from `docs/predictions.md` so that file can be filled in without conflict.

Same 20 scenes (val block 11), same camera, same 40 keyframes per scene, same
LiDAR points and rules. VGGT-Omega sees 640x400 images; VGGT sees 518x322, so
each is scored on ground truth built at its own input size.
`difference = VGGT-Omega minus VGGT`, paired by scene.

Nothing about VGGT on this data has been seen by anyone yet.

---

## Abhishek's predictions

1. Which model has the lower headline AbsRel, and by roughly how much: close 6%
2. Is the gap larger on moving objects than on static background: yes 
3. Which model has the better camera pose (rotation, translation direction, AUC@30): vggt omega 
4. Does VGGT also predict a focal length that is too short: probably not
5. In how many of the 20 scenes will VGGT-Omega have the lower AbsRel: around 12

---

## Claude's predictions (registered 2026-09-18, before VGGT was run on any AURA scene)

1. VGGT-Omega has the lower headline AbsRel, by **0.01 to 0.04** (that is 10 to
   30% relative), and the interval over scenes **excludes zero**. The authors
   report larger relative gains on their own benchmarks; I expect less here,
   because driving depth is dominated by a road plane and large facades that
   both models should handle.
2. The gap is **larger on moving objects** than on background, but the
   moving-object interval will be too wide to exclude zero with so few
   qualifying scenes.
3. Pose: VGGT-Omega better on AUC@30 by **2 to 8 points**; rotation error close
   to a tie, both under 1 degree. VGGT-Omega's DyCheck and TUM-Dynamic gains in
   the authors' table suggest more robustness to moving content.
4. Yes. VGGT also assumes square pixels and a centred principal point, so it
   cannot match this calibration either. I expect **the same sign and a
   similar size** of focal error, within 3 percentage points of VGGT-Omega's.
5. VGGT-Omega lower in **15 or more of 20** scenes.

If VGGT turns out equal or better on this block, that is a real finding for a
post-release dataset and the writeup must lead with it. The one thing it would
NOT show is anything about the paper's scaling claim: two checkpoints that
differ in many ways at once cannot isolate scale.
