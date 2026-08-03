import numpy as np
import pytest

from kai.gui import (
    Hyperparameters,
    TrainingRequest,
    TrainingResult,
    build_results_report,
    save_results_report,
)


def _make_result(standardize: bool = False) -> TrainingResult:
    request = TrainingRequest(
        csv_path=r"C:\data\publicidade.csv",
        label_column="vendas",
        features=("TV", "radio"),
        hyperparameters=Hyperparameters(
            learning_rate=0.05, batch_size=100, epochs=10_000,
            tolerance=1e-6, standardize_features=standardize,
        ),
    )
    return TrainingResult(
        request=request,
        loss_history=(100.0, 20.0, 12.5),
        x_train=np.array([[10.0, 5.0], [20.0, 8.0], [30.0, 11.0], [40.0, 14.0]]),
        y_true=np.array([1.0, 2.0, 3.0, 4.0]),
        y_pred=np.array([1.0, 2.0, 3.0, 4.0]),
        weights=np.array([2.0, 3.0]),
        bias=-0.5,
        elapsed_seconds=2.34,
    )


def test_report_contains_dataset_and_selection_info():
    report = build_results_report(_make_result())
    assert "publicidade.csv" in report
    assert "vendas" in report
    assert "TV, radio" in report


def test_report_contains_training_summary():
    report = build_results_report(_make_result())
    assert "epochs run      : 3" in report
    assert "final MSE       : 12.500000" in report
    assert "00:02:34" in report  # 2.34s as mm:ss:cc


def test_report_contains_model_equation_with_negative_bias():
    report = build_results_report(_make_result())
    assert "vendas = 2.000000 * TV + 3.000000 * radio - 0.500000" in report


def test_report_contains_all_metric_names():
    report = build_results_report(_make_result())
    for name in ("Loss (L1)", "Squared Loss (L2)", "MSE", "RMSE", "R²", "R² Adjusted"):
        assert name in report


def test_report_mentions_standardization_only_when_enabled():
    assert "standardized feature space" in build_results_report(_make_result(standardize=True))
    assert "standardized feature space" not in build_results_report(_make_result(standardize=False))
    assert "standardization : disabled" in build_results_report(_make_result(standardize=False))


def test_save_results_report_writes_utf8_file(tmp_path):
    result = _make_result()
    target = tmp_path / "report.txt"

    saved = save_results_report(result, target)

    assert saved == target
    content = target.read_text(encoding="utf-8")
    assert "kai - Training Report" in content
    assert "publicidade.csv" in content
    assert "R²" in content  # utf-8 round-trip of non-ascii


def test_result_properties():
    result = _make_result()
    assert result.epochs_run == 3
    assert result.final_loss == pytest.approx(12.5)
