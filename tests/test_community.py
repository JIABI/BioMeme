"""Small synthetic checks of the prepared-data adapter; no new scientific analysis."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import sys
import subprocess
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biomeme.community import (SCHEMA, ResponseMemory, load_bundle, save_bundle,
                               ranking_rows, contribution_rows, save_prediction, load_prediction)


def bundles():
    rng = np.random.default_rng(7331)
    meta = dict(schema=SCHEMA, coordinate_system_id="synthetic-frozen-key", output_space_id="synthetic-four-features",
                baseline_id="fixed-synthetic-backbone", target_cell="test", readout="test",
                query_targets_used_in_preparation=False, residual_definition="measured_target_minus_frozen_backbone",
                calibration_residuals="whole_compound_out_of_fold")
    def make(n, start, kind):
        a = dict(ids=np.asarray([f"experiment-{i}" for i in range(start, start+n)]),
                 groups=np.asarray([f"compound-{i}" for i in range(start, start+n)]),
                 names=np.asarray([f"Named experiment {i}" for i in range(start, start+n)]),
                 conditions=np.asarray(["24h-step1"]*n), feature_names=np.asarray(["a", "b", "c", "d"]),
                 keys=rng.normal(size=(2, n, 39)))
        if kind == "query":
            a.update(baseline=rng.normal(size=(n,4)), dmso_center=np.zeros(4))
        else:
            a.update(residuals=rng.normal(size=(n,4)), calibration=np.asarray([kind == "training" and i < 4 for i in range(n)]))
        return a, dict(meta, kind=kind)
    return make(8, 0, "training"), make(3, 30, "query"), make(2, 20, "update")


def test_memory_matches_frozen_predictor_and_named_contributions():
    train, query, _ = bundles()
    m = ResponseMemory.fit(*train)
    result = m.predict(*query)
    direct = query[0]["baseline"] + np.mean([model.predict(z) for model,z in zip(m.models,query[0]["keys"])],axis=0)
    np.testing.assert_allclose(result["prediction"],direct,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(result["weights"].sum(1),1.,rtol=0,atol=1e-12)
    np.testing.assert_allclose(result["baseline"]+result["weights"]@m.arrays["residuals"],direct,rtol=1e-11,atol=1e-11)
    rows = contribution_rows(result,m,query_id=query[0]["ids"][0])
    assert {r['experiment_name'] for r in rows} == set(train[0]["names"])
    with pytest.raises(ValueError,match="different memory"):
        contribution_rows(dict(result,memory_fingerprint=np.asarray("different")),m)
    assert np.isfinite(result["latent_memory_variance"]).all()
    assert (result["latent_memory_variance"] >= 0).all()


def test_update_matches_fresh_fit_and_preserves_previous_model(tmp_path):
    train, query, update = bundles()
    memory=ResponseMemory.fit(*train)
    before=memory.predict(*query)
    changed=memory.update(*update)
    direct=ResponseMemory.fit(changed.arrays,changed.metadata)
    np.testing.assert_allclose(changed.predict(*query)["prediction"],direct.predict(*query)["prediction"],rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(memory.predict(*query)["prediction"],before["prediction"])
    assert [m.geometry for m in memory.models] == [m.geometry for m in changed.models]
    changed.save(tmp_path/"memory")
    loaded=ResponseMemory.load(tmp_path/"memory")
    np.testing.assert_array_equal(loaded.predict(*query)["prediction"],changed.predict(*query)["prediction"])
    save_prediction(tmp_path/"prediction",loaded.predict(*query),loaded)
    np.testing.assert_array_equal(load_prediction(tmp_path/"prediction")["prediction"],loaded.predict(*query)["prediction"])


def test_query_outcome_rejected_before_decoding_arrays(tmp_path,monkeypatch):
    _,query,_=bundles()
    save_bundle(tmp_path/"query",*query,"query")
    class HeaderOnly:
        files=["ids","target"]
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def __getitem__(self,key):raise AssertionError("An outcome array was decoded")
    monkeypatch.setattr(np,"load",lambda *a,**kw:HeaderOnly())
    with pytest.raises(ValueError,match="Unexpected fields"):
        load_bundle(tmp_path/"query","query")


def test_incompatible_spaces_duplicate_updates_and_query_overlap_fail():
    train,query,update=bundles()
    m=ResponseMemory.fit(*train)
    with pytest.raises(ValueError,match="Mismatched coordinate_system_id"):
        m.predict(query[0],dict(query[1],coordinate_system_id="changed"))
    bad={**query[0],"groups":np.asarray([train[0]["groups"][0]]*3)}
    with pytest.raises(ValueError,match="overlap"):
        m.predict(bad,query[1])
    repeated={**update[0],"ids":train[0]["ids"][:2]}
    with pytest.raises(ValueError,match="repeats"):
        m.update(repeated,update[1])
    with pytest.raises(ValueError,match="query_targets_used_in_preparation"):
        ResponseMemory.fit(train[0],dict(train[1],query_targets_used_in_preparation=True))


def test_rankings_use_condition_groups_and_stable_id_ties():
    result=dict(ids=np.asarray(["q3","q2","q1","q4"]),groups=np.asarray(["c3","c2","c1","c4"]),
                names=np.asarray(["3","2","1","4"]),conditions=np.asarray(["a","b","a","b"]),
                latent_memory_variance=np.asarray([2.,100.,2.,3.]),predicted_response_strength=np.asarray([1.,2.,3.,4.]))
    rows=ranking_rows(result,"uncertainty")
    assert [(r['condition'],r['query_id'],r['rank']) for r in rows] == [('a','q1',1),('a','q3',2),('b','q2',1),('b','q4',2)]
    assert [r['query_id'] for r in ranking_rows(result,"activity")] == ['q1','q3','q4','q2']
    del result['predicted_response_strength']
    with pytest.raises(ValueError,match="DMSO"):
        ranking_rows(result,"activity")


def test_complete_command_line_workflow(tmp_path):
    train,query,update=bundles()
    for prefix,bundle,kind in [('train',train,'training'),('query',query,'query'),('new',update,'update')]:
        save_bundle(tmp_path/prefix,*bundle,kind)
    entry=Path(__file__).resolve().parents[1]/'run_memory.py'
    def run(*args):
        subprocess.run([sys.executable,str(entry),*map(str,args)],check=True,capture_output=True,text=True,cwd=tmp_path)
    run('fit','--data',tmp_path/'train','--output',tmp_path/'v1')
    run('predict','--memory',tmp_path/'v1','--data',tmp_path/'query','--output',tmp_path/'p1')
    run('update','--memory',tmp_path/'v1','--data',tmp_path/'new','--output',tmp_path/'v2')
    run('predict','--memory',tmp_path/'v2','--data',tmp_path/'query','--output',tmp_path/'p2')
    run('trace','--memory',tmp_path/'v2','--prediction',tmp_path/'p2','--query-id',query[0]['ids'][0],'--top',3,'--output',tmp_path/'trace.csv')
    run('rank','--prediction',tmp_path/'p2','--by','activity','--output',tmp_path/'ranking.csv')
    assert len((tmp_path/'trace.csv').read_text().splitlines()) == 4
    assert len((tmp_path/'ranking.csv').read_text().splitlines()) == 4
    assert not np.array_equal(load_prediction(tmp_path/'p1')['prediction'],load_prediction(tmp_path/'p2')['prediction'])
