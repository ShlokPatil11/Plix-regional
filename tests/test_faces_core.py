import numpy as np
import pytest

from src.common import PipelineError
from src.faces import (
    FaceCandidate,
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


def test_no_observations_gives_no_candidates():
    assert rank_candidates(cluster_observations([])) == []


def test_lock_tag_and_enabled():
    disabled = FaceLock.disabled()
    assert not disabled.enabled and disabled.tag == "NOFACE"
    lock = FaceLock.from_candidate({"embedding": identity(1).tolist()})
    other = FaceLock.from_candidate({"embedding": identity(2).tolist()})
    assert lock.enabled and lock.tag.startswith("L") and lock.tag != other.tag
    assert lock.vector().shape == (128,)


def test_select_lock_and_unknown_choice(tmp_path):
    candidates = rank_candidates(cluster_observations([obs(identity(1), 0, weight=2.0)]))
    assert select_lock(candidates, "0").enabled
    with pytest.raises(PipelineError):
        select_lock(candidates, "7")


def test_candidates_round_trip_without_losing_embedding(tmp_path):
    candidates = rank_candidates(cluster_observations([obs(identity(1), 0, weight=2.0)]))
    path = tmp_path / "face_candidates.json"
    save_candidates(path, candidates)
    loaded = load_candidates(path)
    assert loaded[0].id == "0" and loaded[0].recommended
    assert cosine(loaded[0].embedding, candidates[0].embedding) == pytest.approx(1.0)
    view = loaded[0].public_view()
    assert "embedding" not in view and view["id"] == "0"
