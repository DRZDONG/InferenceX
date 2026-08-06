import calc_success_rate as success_rate


def test_load_hardware_labels_uses_cluster_labels():
    labels = success_rate.load_hardware_labels()

    assert "b300-nv" in labels
    assert "b300-cw" in labels
    assert "gb300-nv" in labels
    assert "b300" not in labels
    assert all(not label.startswith("cluster:") for label in labels)


def test_extract_hardware_from_name_matches_cluster_label():
    patterns = success_rate.build_hardware_match_patterns(["b300-nv", "gb200-nv"])

    assert (
        success_rate.extract_hardware_from_name(
            "dsv4 fp4 cluster:b300-nv vllm | tp=8", patterns
        )
        == "b300-nv"
    )
    assert (
        success_rate.extract_hardware_from_name(
            "glm5 fp4 gb200-nv dynamo-sglang", patterns
        )
        == "gb200-nv"
    )


def test_extract_hardware_from_name_does_not_infer_broad_sku():
    patterns = success_rate.build_hardware_match_patterns(["b300-nv", "h200-dgxc"])

    assert success_rate.extract_hardware_from_name("dsv4 fp4 b300 vllm", patterns) is None
    assert success_rate.extract_hardware_from_name("dsv4 fp8 h200 sglang", patterns) is None


def test_calculate_success_rates_uses_repository_scoped_token(monkeypatch):
    class FakeRun:
        id = 123
        name = "End-to-End Tests"

        def jobs(self, _filter):
            assert _filter == "all"
            return []

    class FakeRepo:
        full_name = "SemiAnalysisAI/InferenceX-Private-TPU"

        def get_workflow_run(self, run_id):
            assert run_id == 123
            return FakeRun()

    class FakeGithub:
        def __init__(self, auth):
            self.auth = auth

        def get_repo(self, repo_name):
            assert repo_name == "SemiAnalysisAI/InferenceX-Private-TPU"
            return FakeRepo()

    import github

    monkeypatch.setattr(github, "Github", FakeGithub)
    monkeypatch.setattr(success_rate, "GITHUB_TOKEN", "github-actions-token")
    monkeypatch.setattr(success_rate, "REPO_NAME", "SemiAnalysisAI/InferenceX-Private-TPU")
    monkeypatch.setattr(success_rate, "RUN_ID", "123")

    assert success_rate.calculate_hardware_success_rates() == {
        hardware: {"n_success": 0, "total": 0}
        for hardware in success_rate.HARDWARE_LABELS
    }
