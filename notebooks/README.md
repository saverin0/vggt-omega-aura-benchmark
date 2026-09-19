# notebooks/

Every notebook keeps the output of the run that produced the project's numbers; that output is the record.
They run on Colab from a local VS Code. Before running one, put the code into its sync cell:
`python -m vggt_aura.sync` (they are committed with that cell empty, see `src/vggt_aura/sync.py`).

## The run (this folder), in order

| notebook | server | what it does |
|---|---|---|
| `01_census` | CPU | counts scenes per road type, weather and light for every block, from the small base layer; used to choose blocks |
| `02_predict_gpu` | GPU | both models predict every scene of the chosen blocks; predictions are saved |
| `03_score` | CPU, high RAM | ground truth from LiDAR, moving-object labels, scores; one result file per block |
| `03_score_second_server` | CPU, high RAM | optional: the same list walked backwards on a second server |
| `04_report` | CPU | every table, from the saved results only |
| `05_test_predict_gpu`, `06_test_score`, `06_test_score_second_server` | as above | the same steps for the test split, which was kept untouched until the rules were fixed |
| `07_test_report` | CPU | the test split on its own, and against the other scenes inside the same weather and light |
| `08_error_pictures` | CPU | pictures and per-frame tables of single scenes (the committed copy has the pictures removed, see `results/figures/`) |
| `09_moving_object_edges` | CPU, high RAM | is the moving-object error inside the objects or on their outlines? |

## `development/`: how the method was built

`00_environment_probes` and `01_storage_and_session` check the environment and persistent storage.
`02_dataset_inspection` looks at the dataset. `03_first_forward_pass` runs the model once and tests resuming.
`04_ground_truth` builds the LiDAR ground truth and is where the first occlusion rule was found to delete the far
road. `05a_one_block_predict` to `05d_one_block_metrics_vggt` score one block (val 11, the development block) for
both models. `06_cpp_core` builds the C++ core, runs the differential tests on the runtime and times it against
Python.

## `experiments/`: measurements behind the speed-ups

`drive_cache_speed_test`: would a copy of a block on Google Drive beat downloading it again? (not adopted).
`parallel_scoring_identical`: scoring scenes in parallel gives IDENTICAL numbers, 4.8 times faster.
`parallel_unpack_speed_test`: LiDAR archives decompressed on all cores, 5.3 times faster, byte-identical files.
`rerun_check_1_predict_gpu` and `rerun_check_2_score_and_compare`: one small block (val 12, 7 scenes) run again from an
EMPTY folder, then compared with the published run: the predictions, the ground truth and every result number.

## Working names

`docs/decisions.md` was written while the work went on and uses the names the notebooks had then. Text printed
in saved outputs does too.

| then | now |
|---|---|
| `00_probes`, `01_bootstrap`, `02_download_inspect` | `development/00_environment_probes`, `01_storage_and_session`, `02_dataset_inspection` |
| `05a_predict_block`, `05b_metrics`, `05c_predict_block_vggt`, `05d_metrics_vggt` | `development/05a_one_block_predict`, `05b_one_block_metrics`, `05c_one_block_predict_vggt`, `05d_one_block_metrics_vggt` |
| `07a_census` | `01_census` |
| `07b0_predict_blocks_gpu` | `02_predict_gpu` |
| `07b_blocks`, `07b_blocks_second_server` | `03_score`, `03_score_second_server` |
| `07c_report` | `04_report` |
| `07d_cache_speed_test`, `07e_verify_parallel`, `07f_unpack_speed_test` | `experiments/drive_cache_speed_test`, `parallel_scoring_identical`, `parallel_unpack_speed_test` |
| `08a_test_predict_gpu`, `08b_test_score`, `08c_test_report` | `05_test_predict_gpu`, `06_test_score`, `07_test_report` |
| `09_error_pictures`, `10_moving_object_edges` | `08_error_pictures`, `09_moving_object_edges` |
