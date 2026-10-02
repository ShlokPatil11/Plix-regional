import numpy as np
import pytest

from src.common import PipelineError
from src.faces import (
    FaceLock,
    FaceObs,
    cluster_observations,
    cosine,
    load_candidates,
    rank_candidates,
    save_candidates,
    select_lock,
)

rng = np.random.default_rng(7)


def identity(seed):
    return np.random.default_rng(seed).normal(size=128).astype(np.float32)


def noisy(vec, scale=0.05):
    return vec + rng.normal(scale=scale, size=vec.shape).astype(np.float32)


def obs(vec, shot=0, weight=1.0, height=60, frame_h=100):
    return FaceObs(shot_index=shot, time_s=0.0, box=(0, 0, 40, height), frame_h=frame_h,
                   embedding=noisy(vec), weight_s=weight)


def test_cosine():
    a = identity(1)
    assert cosine(a, a) == pytest.approx(1.0)
    assert abs(cosine(identity(1), identity(2))) < 0.4


def test_clustering_groups_same_person_and_separates_others():
    a, b = identity(1), identity(2)
    observations = [obs(a, 0), obs(b, 1), obs(a, 2), obs(b, 3), obs(a, 4)]
    clusters = cluster_observations(observations)
    assert sorted(len(c) for c in clusters) == [2, 3]


def test_ranking_orders_by_screen_time_times_size_and_marks_recommended():
    a, b = identity(1), identity(2)
    clusters = cluster_observations([obs(a, 0, weight=3.0), obs(a, 1, weight=3.0), obs(b, 2, weight=1.0)])
    ranked = rank_candidates(clusters)
    assert [c.id for c in ranked] == ["0", "1"]
    assert ranked[0].recommended and not ranked[1].recommended
    assert ranked[0].shots == [0, 1] and ranked[0].screen_seconds == 6.0


def test_ranking_drops_noise_and_caps_candidates():
    people = [identity(seed) for seed in range(10, 16)]
    clusters = cluster_observations([obs(p, i, weight=2.0 + i) for i, p in enumerate(people)]
                                    + [obs(identity(99), 9, weight=0.05)])
    ranked = rank_candidates(clusters, max_candidates=4)
    assert len(ranked) == 4
    assert all(c.screen_seconds >= 0.3 for c in ranked)


def test_ranking_filters_by_min_seconds():
    # Test that min_seconds filter actually removes sub-threshold candidates
    # One real person (weight 2.0) + one noise identity (weight 0.05)
    clusters = cluster_observations([obs(identity(1), 0, weight=2.0),
                                     obs(identity(99), 1, weight=0.05)])
    ranked = rank_candidates(clusters, max_candidates=4)
    assert len(ranked) == 1  # Only the 2.0s candidate passes min_seconds=0.3 threshold
    assert ranked[0].screen_seconds == 2.0


def test_ranking_prefers_large_faces_over_high_weight():
    # Test that score = weight * (height/frame_h) prefers larger faces over high weight alone
    # Identity X: weight 2.5, height 10 of 100 -> score = 2.5 * 0.1 = 0.25 (high weight, small face)
    # Identity Y: weight 0.5, height 100 of 100 -> score = 0.5 * 1.0 = 0.5 (low weight, large face)
    # Y should rank first (score 0.5) despite much lower weight, because face size matters
    # If height factor were missing, X would rank first with score 2.5 (by weight alone)
    clusters = cluster_observations([obs(identity(1), 0, weight=2.5, height=10),
                                     obs(identity(2), 1, weight=0.5, height=100)])
    ranked = rank_candidates(clusters)
    assert [c.id for c in ranked] == ["0", "1"]
    assert ranked[0].recommended  # First ranked is recommended
    # Verify the scoring: top candidate must have score ~0.5 (large face), not 2.5 (high weight)
    # This fails if the (height/frame_h) factor is missing from the score calculation
    assert ranked[0].score == pytest.approx(0.5, rel=0.01)
    assert ranked[1].score == pytest.approx(0.25, rel=0.01)


def test_no_observations_gives_no_candidates():
    assert rank_candidates(cluster_observations([])) == []


def test_lock_tag_and_enabled():
    disabled = FaceLock.disabled()
    assert not disabled.enabled and disabled.tag == "NOFACE"
    lock = FaceLock.from_candidate({"embedding": identity(1).tolist()})
    other = FaceLock.from_candidate({"embedding": identity(2).tolist()})
    assert lock.enabled and lock.tag.startswith("L") and lock.tag != other.tag
    assert lock.vector().shape == (128,)


def test_select_lock_and_unknown_choice():
    candidates = rank_candidates(cluster_observations([obs(identity(1), 0, weight=2.0)]))
    assert select_lock(candidates, "0").enabled
    with pytest.raises(PipelineError, match="not one of the detected candidates"):
        select_lock(candidates, "7")


def test_candidates_round_trip_without_losing_embedding(tmp_path):
    original = rank_candidates(cluster_observations([obs(identity(1), 0, weight=2.0)]))
    path = tmp_path / "face_candidates.json"
    save_candidates(path, original)
    loaded = load_candidates(path)

    # Verify all fields survive round-trip
    assert loaded[0].id == "0"
    assert loaded[0].recommended == original[0].recommended
    assert loaded[0].shots == original[0].shots
    assert loaded[0].screen_seconds == original[0].screen_seconds
    assert loaded[0].score == original[0].score
    assert loaded[0].crop_file == original[0].crop_file

    # Verify embedding accuracy and stability
    np.testing.assert_allclose(loaded[0].embedding, original[0].embedding, rtol=1e-5)
    assert cosine(loaded[0].embedding, original[0].embedding) == pytest.approx(1.0)

    # Verify tag stability (critical for render phase)
    assert select_lock(loaded, "0").tag == select_lock(original, "0").tag

    # Verify public view excludes embedding
    view = loaded[0].public_view()
    assert "embedding" not in view and view["id"] == "0"
