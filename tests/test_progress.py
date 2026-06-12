from rufpbench.progress import RunProgress, STEP_LABELS


def test_run_progress_parallel_desc_includes_round_and_step():
    progress = RunProgress(enabled=True)
    progress.round_begin(2, 5, generated=10, accepted=1, fp=1, seeds=3)
    progress.seed_begin(2, 3, seed_id="seed_abc", category="metaphor")

    desc = progress.parallel_desc("scout", 4)
    assert "R2/5" in desc
    assert "seed 2/3" in desc
    assert STEP_LABELS["scout"] in desc
    assert "(4)" in desc


def test_run_progress_task_label_for_target_pair():
    class FakeCandidate:
        prompt_id = "p_1234567890abcdef"

    progress = RunProgress(enabled=True)
    label = progress.task_label("scout", (FakeCandidate(), "minimax-m2.7"))
    assert "minimax-m2.7" in label
    assert "p_1234567890ab" in label
