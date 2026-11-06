"""Acceptance tests of saved real-fit outputs and archived statistics."""
from pathlib import Path
import json,os,sys
import pytest
import numpy as np
D=Path(os.environ.get('BIOMEME_DEMO_RESULTS','results/demo'))
S=Path(os.environ.get('BIOMEME_STATS_RESULTS','results/statistics'))
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from biomeme.validation import compare_source_residuals

def test_complete_population_and_archived_predictions():
    r=json.loads((D/'validation.json').read_text());assert r['status']=='passed'
    assert (r['query_compounds'],r['query_rows'],r['outer_folds'])==(101,404,4)
    for f in r['folds']:
        assert max(f['archive_prediction_relative_errors'].values())<=1e-8
        with np.load(ROOT/'data/u2os_demo/archived_source_residuals.npz',allow_pickle=False) as archived, np.load(D/f"fold{f['outer_fold']}/memory_weights.npz",allow_pickle=False) as fitted:
            n=len(archived['ids'])
            np.testing.assert_array_equal(fitted['memory_ids'][:n],archived['ids'])
            check=compare_source_residuals(fitted['training_residuals'][:n],archived['residuals'])
            assert check['relative_l2_error']<=1e-10
            assert check['maximum_absolute_difference']<=1e-10
        assert f['prediction_sealed_before_scoring'] and not f['query_outcomes_used_for_fit']

def test_four_term_reconstruction():
    for f in range(4):
        with np.load(D/f'fold{f}/prediction.npz',allow_pickle=False) as a:
            np.testing.assert_allclose(a['four_terms'].sum(0),a['prediction'],rtol=1e-10,atol=1e-10)

def test_signed_weights_reconstruct_memory_and_sum_one():
    for f in range(4):
        with np.load(D/f'fold{f}/memory_weights.npz',allow_pickle=False) as a,np.load(D/f'fold{f}/prediction.npz',allow_pickle=False) as b:
            np.testing.assert_allclose(a['mean_weights'].sum(1),1,rtol=0,atol=1e-10)
            np.testing.assert_allclose(a['mean_weights']@a['training_residuals'],b['memory_prediction']-b['backbone'],rtol=1e-8,atol=1e-8)
            assert np.isfinite(a['member_latent_krr_variance']).all() and (a['member_latent_krr_variance']>=0).all()

def test_original_nine_direction_tests_and_thirty_loss_tests():
    r=json.loads((S/'core/validation.json').read_text())
    assert r['status']=='passed' and r['comparisons']==30
    assert r['original_direction_tests']==r['direction_tests_favourable']==9
    assert r['bootstrap_matrices_identical'] and r['maximum_absolute_difference_from_archive']<=1e-10

def test_a5_replay():
    r=json.loads((S/'a5/REPLAY_VERIFICATION.json').read_text())
    assert r['passed'] and r['bootstrap_indices_identical'] and r['max_statistic_absolute_error']<=1e-12


def test_residual_comparison_accepts_float_roundoff_without_hash_equality():
    import hashlib
    with np.load(ROOT/'data/u2os_demo/archived_source_residuals.npz',allow_pickle=False) as a:
        archived=a['residuals']
    rounded=np.nextafter(archived,np.inf)
    assert hashlib.sha256(rounded.tobytes()).digest()!=hashlib.sha256(archived.tobytes()).digest()
    assert compare_source_residuals(rounded,archived)['passed']


def test_residual_comparison_rejects_local_corruption():
    with np.load(ROOT/'data/u2os_demo/archived_source_residuals.npz',allow_pickle=False) as a:
        archived=a['residuals']
    corrupted=archived.copy();corrupted[0,0]+=1e-7
    assert np.linalg.norm(corrupted-archived)/np.linalg.norm(archived)<=1e-10
    with pytest.raises(AssertionError,match='beyond tolerance'):
        compare_source_residuals(corrupted,archived)


def test_residual_comparison_rejects_invalid_shape_or_values():
    archived=np.ones((3,4))
    with pytest.raises(ValueError):compare_source_residuals(archived[:2],archived)
    invalid=archived.copy();invalid[0,0]=np.nan
    with pytest.raises(ValueError):compare_source_residuals(invalid,archived)
