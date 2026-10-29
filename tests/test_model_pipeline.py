import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.predict import (ModelNotFoundError, build_custom_row, load_bundle, model_columns,
                         predict_custom, predict_frame, validate_custom_input)
from tests.conftest import needs_artifacts


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


@pytest.fixture(scope="module")
def data():
    return pd.read_parquet(C.SNAPSHOTS_PATH), pd.read_parquet(C.PREDICTIONS_PATH)


def valid_input(**kw):
    base = dict(age=24, height_cm=182, minutes=2400, appearances=30, apps_60plus=27, goals=10,
                assists=5, yellow=3, red=0, euro_minutes_share=0.1, club_ppg=1.8, club_gd_pg=0.6,
                availability=0.7, position="Attack", sub_position="Centre-Forward", league="GB1",
                foot="right")
    base.update(kw); return base


def test_model_not_found_message(tmp_path):
    with pytest.raises(ModelNotFoundError, match="src.train"):
        load_bundle(tmp_path / "nope.joblib")


@needs_artifacts
def test_saved_predictions_match_fresh_inference(bundle, data):
    snaps, preds = data
    test = snaps[snaps.split == "test"].head(300)
    fresh = predict_frame(bundle, test)["predicted_value"].to_numpy()
    saved = preds.set_index(["player_id", "season"]).loc[
        list(zip(test.player_id, test.season)), "predicted_value"].to_numpy()
    assert np.allclose(fresh, saved, rtol=1e-6)


@needs_artifacts
def test_custom_row_reproduces_training_features(bundle, data):
    """Rebuilding features from raw inputs must give the same prediction as the stored row."""
    snaps, _ = data
    r = snaps[(snaps.split == "test") & (snaps.position == "Midfield") & snaps.height_cm.notna()
              & snaps.club_ppg.notna()].iloc[0]
    inp = dict(age=float(r.age), height_cm=float(r.height_cm), minutes=float(r.w_minutes),
               appearances=float(r.appearances), apps_60plus=float(r.apps_60plus),
               goals=float(r.w_goals), assists=float(r.w_assists),
               yellow=float(r.yellow_p90 * r.w_minutes / 90), red=float(r.red_p90 * r.w_minutes / 90),
               euro_minutes_share=float(r.euro_minutes_share), club_ppg=float(r.club_ppg),
               club_gd_pg=float(r.club_gd_pg), availability=float(r.availability),
               position=r.position, sub_position=r.sub_position, league=r.league, foot=r.foot)
    row = build_custom_row(inp, bundle)
    assert list(row.columns) == model_columns(bundle)
    stored = r.copy(); stored["market_index"] = bundle["latest_market_index"]
    a = predict_frame(bundle, row)["predicted_value"].iloc[0]
    b = predict_frame(bundle, stored.to_frame().T.infer_objects())["predicted_value"].iloc[0]
    assert a == pytest.approx(b, rel=1e-4)


@needs_artifacts
def test_custom_prediction_and_intervals(bundle):
    res = predict_custom(valid_input(), bundle)
    assert res["pred_lo90"] < res["pred_lo80"] < res["predicted_value"] < res["pred_hi80"] < res["pred_hi90"]
    assert res["predicted_value"] > 0


@needs_artifacts
@pytest.mark.parametrize("bad", [dict(age=10), dict(minutes=100), dict(apps_60plus=40, appearances=30),
                                 dict(position="Striker"), dict(league="XX9"), dict(red=9),
                                 dict(minutes=5000, appearances=10)])
def test_invalid_inputs_are_rejected(bundle, bad):
    errs, _ = validate_custom_input(valid_input(**bad), bundle)
    assert errs
    with pytest.raises(ValueError):
        predict_custom(valid_input(**bad), bundle)


@needs_artifacts
def test_extrapolation_warning(bundle):
    _, warns = validate_custom_input(valid_input(goals=65, minutes=2400, appearances=40, assists=40), bundle)
    assert any("outside the typical training range" in w for w in warns)


@needs_artifacts
def test_missing_columns_in_frame_raise(bundle):
    with pytest.raises(ValueError, match="Missing feature columns"):
        predict_frame(bundle, pd.DataFrame({"age": [20]}))


@needs_artifacts
def test_split_is_chronological_and_has_no_overlap(data):
    snaps, _ = data
    mx = snaps.groupby("split").season.agg(["min", "max"])
    assert mx.loc["train", "max"] < mx.loc["val", "min"] <= mx.loc["val", "max"] < mx.loc["test", "min"]
    assert not snaps.duplicated(["player_id", "season"]).any()
