from gpsi.core import assign_gpsi_stages_by_cluster, select_kmeans_clusters
from gpsi.preprocessing import preprocess_features
from gpsi.simulation import simulate_gpsi_data


def test_simulated_gpsi_smoke():
    data = simulate_gpsi_data(n_subjects=60, n_features=5, seed=7)
    feature_cols = [c for c in data.columns if c.startswith("feature_")]
    X, selected, _ = preprocess_features(data, feature_cols)
    assert X.shape[0] == 60
    assert len(selected) > 0

    k_result = select_kmeans_clusters(X, k_min=2, k_max=3, seed=7, n_init=5)
    assert k_result.best_k in {2, 3}

    stage_result = assign_gpsi_stages_by_cluster(
        X,
        k_result.labels,
        seed=7,
        max_fit_subjects=25,
        n_pca_features=3,
        mh_iterations=20,
        mh_burn_in=5,
    )
    assert stage_result.pseudo_stage.shape[0] == 60
    assert stage_result.stage_bin.min() >= 1
