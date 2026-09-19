# Registered predictions

Written BEFORE the first run of `notebooks/development/05b_one_block_metrics.ipynb` (then named `05b_metrics`). Commit this file
before that run, so the commit timestamp shows the predictions came first.
The notebook refuses to score anything until `PREDICTIONS_REGISTERED = True`.

Scope of the first run: val block 11, 20 scenes, camera front_medium, 40
keyframes per scene, checkpoint `vggt_omega_1b_512.pt`. Primary protocol:
`sequence_scale` (one median scale per sequence). Depth range 1 to 80 m.

What was already seen, and is therefore NOT a clean prediction: `03_first_forward_pass` looked
at one scene (`2025-06-13-07-09-37|78`). Its depth maps looked coherent, its
predicted camera path had the right shape, and its focal length came out 8%
short in x and 12% short in y. No depth error number has been computed for any
scene.

---

## Abhishek's predictions

Fill in a number or a direction for each, and one line of reasoning. "I don't
know" is a valid answer; write it down rather than leaving a blank.

1. Headline AbsRel, all pixels, `sequence_scale`: around 0.15, driving scenes are hard
2. How much lower is AbsRel under the authors' per-frame scale-and-shift: a little
3. Moving-object pixels vs static background (direction, rough size): moving will be worse, maybe double
4. Parked-object pixels vs static background: motion is the problem no the car
5. AbsRel at 40 to 80 m compared with 1 to 10 m: yes certainly, 2 times
6. Is the model's confidence informative? AbsRel of its least-sure quarter vs its most-sure quarter: not entirely. least-sure quarter
7. Best and worst semantic group (flat, vehicle, human, structure, thin, vegetation): vegetation is best and roads are worst
8. Pose: median rotation error, median translation-direction error, AUC@30: I don't know
9. Do depth and camera translation agree on the scale (ratio near 1)? between 1 and 2
10. Near-stationary scenes (under 5 km/h): depth better, same, or worse than moving scenes: better
11. Wet vs dry scenes (only 3 wet scenes, so descriptive at best): yes 
12. Which single scene will be worst, and why:rainy one as camera wouldnt able to detect pixels better

---

## Claude's predictions (registered 2026-09-18, before any depth error was computed)

A second forecaster, so there is something to compare against. Intervals are
my rough 80% ranges.

1. Headline AbsRel, `sequence_scale`: **0.10** (0.06 to 0.16). The authors report
   0.04 to 0.06 on their outdoor-ish benchmarks, but under per-frame
   scale-and-shift and on denser, nearer ground truth. Driving has long depth
   ranges and 38% of our pixels are vegetation.
2. Authors' scale-and-shift vs one scale per sequence: **20 to 40% lower AbsRel**.
   Most of the gap from scale drift along the sequence, which one scale per
   sequence does not forgive.
3. Moving objects vs background: **worse, by +0.03 to +0.10 AbsRel**, in most scenes
   that have enough moving pixels. But I expect FEWER THAN 10 of 20 scenes to
   qualify (200+ moving pixels), so the interval will be wide. The trap-3 bound
   on the ground truth itself: median 1 to 3% of depth, which is NOT negligible
   next to the effect.
4. Parked objects vs background: **slightly worse, +0.01 to +0.04**. Cars are small,
   have sharp outlines, and boundary pixels hurt. Much smaller than item 3.
5. Depth bands: error rises with distance. **40 to 80 m at least 2x the 1 to 10 m
   AbsRel.**
6. Confidence is informative: **least-sure quarter at least 2x the AbsRel of the
   most-sure quarter.** `03_first_forward_pass` showed low confidence on outlines and thin
   structures, which is where errors live.
7. Best group **flat** (road, sidewalk). Worst group **thin** (poles, signs), then
   vegetation.
8. Pose on scenes that move: median rotation error **under 1 degree**, median
   translation-direction error **under 3 degrees**, AUC@30 **above 85**, ATE after
   scale-only alignment **under 3% of path length**. Forward driving is an easy
   trajectory shape even though parallax is poor.
9. Scale agreement: ratio **between 0.95 and 1.05**. Both heads are trained on
   one shared normalisation.
10. Near-stationary scenes: depth **about the same or slightly better**. Depth
    here is driven mostly by single-image cues, not parallax. Their pose is
    excluded by rule, so this is a depth-only claim.
11. Wet vs dry: **no visible difference** at n = 3. Light rain on a daytime
    street changes little for the camera.
12. Focal length (partly seen already, see above): fx **5 to 10% short**, fy
    **10 to 14% short**, in every scene, because the model assumes square pixels
    and this calibration is not square.
13. Occlusion rule sensitivity: background AbsRel moves by **under 0.005**
    between `none` and `two_sided`; object strata move more.

The prediction I am least sure of is item 3's size. If moving objects are NOT
worse than parked ones, the "static-scene assumption breaks on moving vehicles"
story in the project brief does not hold for this model on this data, and the
writeup must say so.
