import pandas as pd

from pipeline_v4.src import production_workflow


def test_bounded_batches_preserve_order_and_n_jobs_outputs(monkeypatch, tmp_path):
    gps = pd.DataFrame({
        "caid": [101, 101, 202, 202, 303, 303],
        "local_timestamp": pd.to_datetime([
            "2026-01-01 08:00", "2026-01-02 08:00",
            "2026-01-01 08:00", "2026-01-02 08:00",
            "2026-01-01 08:00", "2026-01-02 08:00",
        ]),
        "latitude": [25.68] * 6, "longitude": [-100.31] * 6,
    })
    metadata = pd.DataFrame({
        "user_id": [101, 202, 303], "routing_eligible": [True] * 3,
    })

    def fake_process(user_id, day_frame, resources):
        day = day_frame.local_timestamp.iloc[0].strftime("%Y-%m-%d")
        physical_trip_id = f"{user_id}_{day}_1"
        ledger = pd.DataFrame([{
            "user_id": user_id, "trip_id": 1, "physical_trip_id": physical_trip_id,
            "processing_status": "quality_rejected", "failure_reason": "fixture",
            "raw_ping_count": 1, "effective_ping_count": 1,
            "pct_pings_conserved": 100.0, "classification_success": False,
            "route_success": False, "emissions_success": False,
        }])
        return production_workflow.DayProcessingResult(pd.DataFrame(), ledger)

    monkeypatch.setattr(production_workflow, "process_user_day", fake_process)
    outputs = []
    for jobs, output in ((1, tmp_path / "jobs1"), (2, tmp_path / "jobs2")):
        result = production_workflow.run_pipeline_v4(
            gps, output, metadata, n_jobs=jobs, resources={"fixture": True},
            output_mode="summary", user_day_batch_size=2,
        )
        outputs.append(result.trip_ledger.copy())
    expected = [
        "101_2026-01-01_1", "101_2026-01-02_1",
        "202_2026-01-01_1", "202_2026-01-02_1",
        "303_2026-01-01_1", "303_2026-01-02_1",
    ]
    assert outputs[0].physical_trip_id.tolist() == expected
    assert outputs[1].physical_trip_id.tolist() == expected
    pd.testing.assert_frame_equal(outputs[0], outputs[1], check_dtype=False)


def test_pipeline_wrapper_uses_routing_eligibility_not_home_quality(monkeypatch, tmp_path):
    gps = pd.DataFrame({
        "caid": [101, 202], "user_id": [101, 202],
        "local_timestamp": pd.to_datetime(["2026-01-01 08:00", "2026-01-01 09:00"]),
        "latitude": [25.68, 25.69], "longitude": [-100.31, -100.30],
    })
    metadata = pd.DataFrame({
        "user_id": [101, 202], "home_lat": [25.67, 25.66],
        "home_lon": [-100.32, -100.33], "home_ageb": ["A", "B"],
        "home_confidence": [.8, .2], "home_quality_flag": ["reliable", "ambiguous"],
        "routing_eligible": [True, True],
        "processing_status": ["ready_for_pipeline", "ready_for_pipeline"],
    })
    calls = []




    def fake_process(user_id, day_frame, resources):
        calls.append(user_id)
        routes = pd.DataFrame({
            "caid": [user_id], "trip": [1], "local_timestamp": [day_frame.local_timestamp.iloc[0]],
            "physical_trip_id": [f"{user_id}_2026-01-01_1"],
            "distance_m": [100.0], "modo_transporte": ["Carro"], "osmid": ["1"],
        })
        ledger = pd.DataFrame([{
            "user_id": user_id, "trip_id": 1,
            "physical_trip_id": f"{user_id}_2026-01-01_1",
            "processing_status": "emissions_pending", "failure_reason": None,
            "raw_ping_count": 20, "effective_ping_count": 20, "pct_pings_conserved": 100.0,
            "hypotheses_attempted": "Carro", "hypotheses_successful": "Carro",
            "hypotheses_attempted_count": 1, "hypotheses_successful_count": 1,
            "route_success": True, "final_mode": "Carro",
            "classification_success": True, "emissions_success": False,
        }])
        return production_workflow.DayProcessingResult(routes, ledger)

    monkeypatch.setattr(production_workflow, "process_user_day", fake_process)
    monkeypatch.setattr(
        production_workflow, "calculate_emissions",
        lambda routes, _: routes.assign(Total_CO2_g=10.0),
    )
    result = production_workflow.run_pipeline_v4(
        gps, tmp_path, metadata, n_jobs=1, resources={"fixture": True}
    )
    assert calls == [101, 202]
    assert result.individual_emissions.user_id.tolist() == [101, 202]
    assert result.individual_emissions.home_ageb.tolist() == ["A", "B"]
    assert (tmp_path / "routes_emissions_summary.parquet").exists()
    assert not (tmp_path / "routes_emissions_detailed.parquet").exists()
    assert (tmp_path / "trip_ledger.parquet").exists()
    assert (tmp_path / "pipeline_manifest.json").exists()
    assert result.output_artifacts["summary_output"]["generated"] is True
    assert result.output_artifacts["detailed_output"]["generated"] is False
    assert len(result.trip_ledger) == 2
    assert result.trip_ledger.iloc[0].processing_status == "success"


def test_global_progress_bar_accounting_and_lifecycle(monkeypatch, tmp_path):
    gps = pd.DataFrame({
        "caid": [101, 101, 202, 202, 303, 303],
        "local_timestamp": pd.to_datetime([
            "2026-01-01 08:00", "2026-01-02 08:00",
            "2026-01-01 08:00", "2026-01-02 08:00",
            "2026-01-01 08:00", "2026-01-02 08:00",
        ]),
        "latitude": [25.68] * 6, "longitude": [-100.31] * 6,
    })
    metadata = pd.DataFrame({
        "user_id": [101, 202, 303], "routing_eligible": [True] * 3,
    })

    def fake_process(user_id, day_frame, resources):
        day = day_frame.local_timestamp.iloc[0].strftime("%Y-%m-%d")
        ledger = pd.DataFrame([{
            "user_id": user_id, "trip_id": 1, "physical_trip_id": f"{user_id}_{day}_1",
            "processing_status": "quality_rejected", "failure_reason": "fixture",
            "raw_ping_count": 1, "effective_ping_count": 1,
            "pct_pings_conserved": 100.0, "classification_success": False,
            "route_success": False, "emissions_success": False,
        }])
        return production_workflow.DayProcessingResult(pd.DataFrame(), ledger)

    monkeypatch.setattr(production_workflow, "process_user_day", fake_process)

    class TrackedProgressBar:
        def __init__(self, total, **kwargs):
            self.total = total
            self.kwargs = kwargs
            self.updates = []
            self.closed = False

        def update(self, n=1):
            self.updates.append(n)

        def close(self):
            self.closed = True

    created_bars = []

    def mock_tqdm(*args, **kwargs):
        bar = TrackedProgressBar(*args, **kwargs)
        created_bars.append(bar)
        return bar

    monkeypatch.setattr(production_workflow, "tqdm", mock_tqdm)

    # 1. Run with show_progress=True and batch_size=2 across 6 user-days
    production_workflow.run_pipeline_v4(
        gps, tmp_path / "run_progress_true", metadata, n_jobs=2,
        resources={"fixture": True}, output_mode="summary",
        user_day_batch_size=2, show_progress=True,
    )
    assert len(created_bars) == 1
    bar = created_bars[0]
    assert bar.total == 6
    assert bar.kwargs.get("desc") == "Processing user-days"
    assert sum(bar.updates) == 6
    assert bar.closed is True

    # 2. Run with show_progress=False
    created_bars.clear()
    production_workflow.run_pipeline_v4(
        gps, tmp_path / "run_progress_false", metadata, n_jobs=2,
        resources={"fixture": True}, output_mode="summary",
        user_day_batch_size=2, show_progress=False,
    )
    assert len(created_bars) == 0
