import json

import numpy as np
import pandas as pd
import pytest

from vggt_aura import aura_data as ad, ground_truth as gtm, inference as inf, pins, pipeline as pl


def test_block_names_sort_correctly():
    assert pl.block_tag("val", 11) == "val_block000011"
    assert sorted([pl.block_tag("val", 100), pl.block_tag("val", 11), pl.block_tag("val", 2)]) == \
        ["val_block000002", "val_block000011", "val_block000100"]


def test_results_are_kept_apart_by_run_model_and_block(tmp_path):
    a = pl.block_result_paths(tmp_path, "run1", "vggt_omega_512", "val", 11)
    b = pl.block_result_paths(tmp_path, "run1", "vggt_1b", "val", 11)
    c = pl.block_result_paths(tmp_path, "run1", "vggt_omega_512", "val", 12)
    assert len({a, b, c}) == 3
    assert a[0].name == "val_block000011_rows.csv" and a[1].name == "val_block000011_scenes.csv"
    assert a[0].parent == tmp_path / "metrics" / "run1" / "vggt_omega_512" / "blocks"


def write_block(tmp_path, model, block, scene_ids, only_rows=False):
    rows_path, scenes_path = pl.block_result_paths(tmp_path, "run1", model, "val", block)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"scene_id": scene_ids, "abs_rel": [0.05] * len(scene_ids)}).to_csv(rows_path, index=False)
    if not only_rows:
        pd.DataFrame({"scene_id": scene_ids, "block": [block] * len(scene_ids)}).to_csv(scenes_path, index=False)


def test_a_block_is_done_only_when_every_model_has_both_files(tmp_path):
    models = ["vggt_omega_512", "vggt_1b"]
    assert not pl.block_is_done(tmp_path, "run1", models, "val", 11)
    write_block(tmp_path, "vggt_omega_512", 11, ["a|1"])
    assert not pl.block_is_done(tmp_path, "run1", models, "val", 11)          # the second model is missing
    write_block(tmp_path, "vggt_1b", 11, ["a|1"], only_rows=True)
    assert not pl.block_is_done(tmp_path, "run1", models, "val", 11)          # a half-written pair does not count
    write_block(tmp_path, "vggt_1b", 11, ["a|1"])
    assert pl.block_is_done(tmp_path, "run1", models, "val", 11)
    assert pl.block_is_done(tmp_path, "run1", ["vggt_omega_512"], "val", 11)


def test_a_run_is_stitched_from_its_blocks_without_any_dataset(tmp_path):
    write_block(tmp_path, "vggt_omega_512", 11, ["a|1", "a|2"])
    write_block(tmp_path, "vggt_omega_512", 2, ["b|1"])
    rows, scenes = pl.load_run(tmp_path, "run1", "vggt_omega_512")
    assert sorted(rows["scene_id"]) == ["a|1", "a|2", "b|1"] and sorted(scenes["block"]) == [2, 11, 11]
    empty_rows, empty_scenes = pl.load_run(tmp_path, "run1", "vggt_1b")
    assert empty_rows.empty and empty_scenes.empty


def test_model_runner_knows_where_each_model_stores_predictions():
    assert pl.ModelRunner("vggt_omega_512").folder == pins.CHECKPOINT_PUBLIC_512
    assert pl.ModelRunner("vggt_1b").folder == "vggt_1b"
    with pytest.raises(ValueError):
        pl.ModelRunner("vggt_2b")


def test_releasing_an_unused_runner_is_harmless():
    pl.ModelRunner("vggt_1b").release()


def test_a_scene_in_two_blocks_is_an_error_not_a_double_count(tmp_path):
    write_block(tmp_path, "vggt_omega_512", 11, ["a|1", "a|2"])
    write_block(tmp_path, "vggt_omega_512", 12, ["a|2"])
    with pytest.raises(ValueError, match="more than one block"):
        pl.load_run(tmp_path, "run1", "vggt_omega_512")


def save_scene(root, name, model, input_hw, with_truth=True):
    npz, meta = inf.prediction_paths(root, name, "front_medium", pl.ModelRunner(model).folder)
    npz.parent.mkdir(parents=True, exist_ok=True)
    npz.write_bytes(b"x")
    meta.write_text(json.dumps({"input_hw": input_hw}), encoding="utf-8")
    if with_truth:
        for path in gtm.truth_paths(root, name, "front_medium", tag=gtm.truth_tag(input_hw[1], input_hw[0])):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")


def test_lidar_is_skipped_only_when_everything_is_saved(tmp_path):
    models, names = ["vggt_omega_512", "vggt_1b"], ["2025-06-13-07-09-37_78", "2025-06-13-07-09-37_39"]
    sizes = {"vggt_omega_512": [400, 640], "vggt_1b": [322, 518]}
    assert not pl.block_is_fully_saved(tmp_path, names, "front_medium", models)          # nothing saved
    assert not pl.block_is_fully_saved(tmp_path, None, "front_medium", models)           # names unknown: never assume
    for name in names:
        for model in models:
            save_scene(tmp_path, name, model, sizes[model])
    assert pl.block_is_fully_saved(tmp_path, names, "front_medium", models)
    # each model's ground truth is looked up at that model's own input size
    assert gtm.truth_paths(tmp_path, names[0], "front_medium", tag="518x322")[0].is_file()
    assert gtm.truth_paths(tmp_path, names[0], "front_medium")[0].is_file()

    gtm.truth_paths(tmp_path, names[1], "front_medium", tag="518x322")[0].unlink()     # one ground truth missing
    assert not pl.block_is_fully_saved(tmp_path, names, "front_medium", models)
    assert pl.block_is_fully_saved(tmp_path, names, "front_medium", ["vggt_omega_512"])  # the other model is complete
    assert not pl.block_is_fully_saved(tmp_path, names + ["a_new_scene_1"], "front_medium", ["vggt_omega_512"])


def test_scene_names_follow_the_same_order_as_scene_ids():
    table = pd.DataFrame({"split": ["val"] * 3, "scene_block": [11, 11, 2], "scene_ordinal": [2, 1, 3],
                          "scene_id": ["d|2", "d|1", "e|9"], "scene": ["d_2", "d_1", "e_9"]})
    assert ad.block_scene_ids(table, "val", 11) == ["d|1", "d|2"]
    assert ad.block_scene_names(table, "val", 11) == ["d_1", "d_2"]


def test_block_has_predictions_needs_every_scene_and_every_model(tmp_path):
    models, names = ["vggt_omega_512", "vggt_1b"], ["d_1", "d_2"]
    assert not pl.block_has_predictions(tmp_path, names, "front_medium", models)
    assert not pl.block_has_predictions(tmp_path, None, "front_medium", models)
    for name in names:
        save_scene(tmp_path, name, "vggt_omega_512", [400, 640], with_truth=False)
    assert pl.block_has_predictions(tmp_path, names, "front_medium", ["vggt_omega_512"])
    assert not pl.block_has_predictions(tmp_path, names, "front_medium", models)      # the second model is missing
    save_scene(tmp_path, "d_1", "vggt_1b", [322, 518], with_truth=False)
    assert not pl.block_has_predictions(tmp_path, names, "front_medium", models)      # one scene still missing
    save_scene(tmp_path, "d_2", "vggt_1b", [322, 518], with_truth=False)
    assert pl.block_has_predictions(tmp_path, names, "front_medium", models)
    assert not pl.block_is_fully_saved(tmp_path, names, "front_medium", models)       # predictions are not ground truth


def test_an_old_census_table_is_recounted_a_current_one_is_kept(tmp_path):
    assert not pl.census_is_current(tmp_path, "val", 11)                       # nothing there yet
    path = pl.census_path(tmp_path, "val", 11)
    path.parent.mkdir(parents=True)
    pd.DataFrame({"scene_id": ["a|1"], "n_lidars": [10], "has_aeva": [True]}).to_csv(path, index=False)
    assert not pl.census_is_current(tmp_path, "val", 11)                       # from before the Ouster column
    pd.DataFrame({"scene_id": ["a|1"], "n_lidars": [10], "has_aeva": [True], "n_lidars_ouster": [6],
                  "lidar_ids": ["aeva_a top_left"]}).to_csv(path, index=False)
    assert pl.census_is_current(tmp_path, "val", 11)


def jobs(n):
    return [{"scene_id": f"s|{k}", "value": k} for k in range(n)]


def test_jobs_come_back_in_order_one_at_a_time():
    seen = []
    results = pl.run_jobs(jobs(5), workers=1, function=pl._selftest_job, on_result=lambda r: seen.append(r["scene_id"]))
    assert [r["value"] for r in results] == [0, 1, 4, 9, 16]
    assert seen == ["s|0", "s|1", "s|2", "s|3", "s|4"]


def test_jobs_come_back_in_the_same_order_from_worker_processes():
    import os
    results = pl.run_jobs(jobs(6), workers=3, function=pl._selftest_job)
    assert [r["scene_id"] for r in results] == [f"s|{k}" for k in range(6)]
    assert [r["value"] for r in results] == [0, 1, 4, 9, 16, 25]
    assert os.getpid() not in {r["pid"] for r in results}                  # the work really left this process


def test_parallel_and_serial_give_identical_results():
    serial = [{k: v for k, v in r.items() if k != "pid"} for r in pl.run_jobs(jobs(7), 1, pl._selftest_job)]
    parallel = [{k: v for k, v in r.items() if k != "pid"} for r in pl.run_jobs(jobs(7), 4, pl._selftest_job)]
    assert serial == parallel


def test_a_failing_scene_is_reported_by_name_not_swallowed():
    broken = jobs(4)
    broken[2]["fail"] = True
    for workers in (1, 2):
        with pytest.raises(RuntimeError, match=r"planted failure in s\|2"):
            pl.run_jobs(broken, workers, pl._selftest_job)


def test_worker_count_is_sane():
    assert 1 <= pl.default_workers() <= 8
    assert pl.run_jobs([], workers=4, function=pl._selftest_job) == []


def test_saved_predictions_are_recognised_by_their_frame_times(tmp_path):
    import numpy as np
    stamps = [1752585602300000000 + k * 500_000_000 for k in range(3)]
    folder = pl.ModelRunner("vggt_omega_512").folder
    assert not pl.predictions_are_saved(tmp_path, "scene_1", "front_medium", folder, stamps)       # nothing saved
    npz, meta = inf.prediction_paths(tmp_path, "scene_1", "front_medium", folder)
    arrays = {"depth": np.ones((3, 4, 6)), "depth_conf": np.ones((3, 4, 6)), "extrinsics": np.zeros((3, 3, 4)),
              "intrinsics": np.zeros((3, 3, 3)), "pose_enc": np.zeros((3, 9)), "input_hw": np.array([4, 6]),
              "gt_camera0_from_camera": np.zeros((3, 4, 4)), "timestamps_ns": np.array(stamps, dtype=np.int64)}
    inf.save_predictions(npz, meta, arrays, {"input_hw": [4, 6]})
    assert pl.predictions_are_saved(tmp_path, "scene_1", "front_medium", folder, stamps)
    assert not pl.predictions_are_saved(tmp_path, "scene_1", "front_medium", folder, stamps[:2])    # other frames
    assert not pl.predictions_are_saved(tmp_path, "scene_1", "front_medium", "vggt_1b", stamps)     # other model
    npz.write_bytes(b"not a zip file")                                                             # a damaged file
    assert not pl.predictions_are_saved(tmp_path, "scene_1", "front_medium", folder, stamps)


def test_scenes_with_too_few_ouster_sensors_are_recognised():
    six = ["front_left", "front_right", "rear_left", "rear_right", "top_left", "top_right"]
    assert pl.enough_ouster(six) and pl.enough_ouster(six + ["aeva_front", "aeva_rear"])
    assert not pl.enough_ouster(six[:4])                                  # the 55 four-sensor train scenes
    assert not pl.enough_ouster(six[:4] + ["aeva_a", "aeva_b", "aeva_c"])    # Aeva sensors do not count
    assert pl.enough_ouster(six[:4], min_ouster=4)


def test_two_servers_share_a_block_list_through_claims(tmp_path):
    assert pl.claim_block(tmp_path, "run", "val", 1, owner="forward")
    assert pl.claim_block(tmp_path, "run", "val", 1, owner="forward")            # asking again is fine
    assert not pl.claim_block(tmp_path, "run", "val", 1, owner="backward")       # the other server leaves it alone
    assert pl.claim_block(tmp_path, "run", "val", 2, owner="backward")           # and takes a different block
    pl.release_claim(tmp_path, "run", "val", 1, owner="backward")                # only the owner can release
    assert not pl.claim_block(tmp_path, "run", "val", 1, owner="backward")
    pl.release_claim(tmp_path, "run", "val", 1, owner="forward")
    assert pl.claim_block(tmp_path, "run", "val", 1, owner="backward")


def test_a_claim_left_by_a_dead_server_is_taken_over(tmp_path):
    assert pl.claim_block(tmp_path, "run", "val", 1, owner="died")
    assert not pl.claim_block(tmp_path, "run", "val", 1, owner="alive", max_age_s=3600)
    assert pl.claim_block(tmp_path, "run", "val", 1, owner="alive", max_age_s=0)
    pl.claim_path(tmp_path, "run", "val", 3).parent.mkdir(parents=True, exist_ok=True)
    pl.claim_path(tmp_path, "run", "val", 3).write_text("half-written {", encoding="utf-8")
    assert pl.claim_block(tmp_path, "run", "val", 3, owner="alive")               # an unreadable claim does not block anyone


def test_prepared_ahead_keeps_the_order_and_really_works_ahead():
    import threading
    import time as clock

    in_flight, most = [0], [0]
    lock = threading.Lock()

    def prepare(job):
        with lock:
            in_flight[0] += 1
            most[0] = max(most[0], in_flight[0])
        clock.sleep(0.05)
        with lock:
            in_flight[0] -= 1
        return job * 10

    started = clock.time()
    seen = []
    for job, prepared, waited in pl.prepared_ahead(range(8), prepare, lookahead=2, threads=2):
        clock.sleep(0.05)                                   # the "GPU" works on this job meanwhile
        seen.append((job, prepared))
    elapsed = clock.time() - started
    assert seen == [(k, k * 10) for k in range(8)]
    assert most[0] <= 2                                     # never more than `lookahead` prepared at once
    assert elapsed < 8 * 0.10 * 0.85                        # faster than doing prepare and work one after the other


def test_prepared_ahead_raises_the_error_at_the_job_it_belongs_to():
    def prepare(job):
        if job == 2:
            raise OSError("unreadable image")
        return job

    seen = []
    with pytest.raises(OSError, match="unreadable"):
        for job, prepared, _ in pl.prepared_ahead(range(5), prepare):
            seen.append(job)
    assert seen == [0, 1]
    assert list(pl.prepared_ahead([], prepare)) == []


def test_warm_files_reads_what_exists_and_ignores_what_does_not(tmp_path):
    for k in range(5):
        (tmp_path / f"p{k}.npz").write_bytes(b"x" * (1000 * (k + 1)))
    report = pl.warm_files([tmp_path / f"p{k}.npz" for k in range(5)] + [tmp_path / "missing.npz"], threads=3)
    assert report["files"] == 5 and report["mb"] == round(15000 / 1e6, 1)
    assert sorted(f.name for f in tmp_path.iterdir()) == [f"p{k}.npz" for k in range(5)]        # nothing written
    assert pl.warm_files([])["files"] == 0


def test_background_returns_the_value_and_re_raises_the_error():
    assert pl._Background(lambda a, b: a + b, 2, 3).result() == 5

    def broken():
        raise ValueError("validator crashed")

    with pytest.raises(ValueError, match="validator crashed"):
        pl._Background(broken).result()


def test_frames_are_found_by_timestamp_whatever_layers_are_on_disk():
    import types

    class Frames(list):                                  # the toolkit's frame list: len and index, no iteration needed here
        pass

    frames = Frames(types.SimpleNamespace(timestamp_ns=1000 * k) for k in range(8))
    scene = types.SimpleNamespace(scene_id="s|1", frames=lambda sample_filter: frames)
    found = pl.frames_for_timestamps(scene, np.array([3000, 4000, 6000], dtype=np.int64))
    assert [f.timestamp_ns for f in found] == [3000, 4000, 6000]
    with pytest.raises(RuntimeError, match="1 of 2"):
        pl.frames_for_timestamps(scene, [1000, 1500])


def test_only_a_missing_ground_truth_triggers_the_second_download(monkeypatch):
    """The camera-only shortcut may retry with LiDAR when a saved ground truth does not fit. Any other failure,
    a rejected download for one, must stop the block instead of being retried under a wrong explanation."""
    import types

    session = types.SimpleNamespace(persist_root="unused")
    monkeypatch.setattr(pl, "block_is_done", lambda *a: False)
    monkeypatch.setattr(pl, "block_is_fully_saved", lambda *a: True)
    calls = []

    def first_fails_with(error):
        def fake(session, split, block, scene_ids, camera_id, models, run_tag, lidar_policy, occlusion, validate,
                 keep_data, layers, *rest):
            calls.append(list(layers))
            if len(calls) == 1:
                raise error
            return {"status": "done", "layers": list(layers)}
        return fake

    monkeypatch.setattr(pl, "_process_block", first_fails_with(gtm.GroundTruthUnavailable("no saved ground truth matches")))
    out = pl.process_block(session, "val", 1, ["s|1"], "front_medium", ["vggt_omega_512"], "run", scene_names=["s_1"])
    assert out["status"] == "done" and calls == [[pl.CAMERA_LAYER], [pl.CAMERA_LAYER, pl.LIDAR_LAYER]]

    calls.clear()
    monkeypatch.setattr(pl, "_process_block", first_fails_with(RuntimeError("the validator rejected the download")))
    with pytest.raises(RuntimeError, match="validator"):
        pl.process_block(session, "val", 1, ["s|1"], "front_medium", ["vggt_omega_512"], "run", scene_names=["s_1"])
    assert calls == [[pl.CAMERA_LAYER]]                                   # no second attempt
