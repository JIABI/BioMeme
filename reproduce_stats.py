#!/usr/bin/env python3
"""Recompute Table 1's original 9/9 directions, key Table 2 rows, and A5."""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):os.environ[key]="1"
from pathlib import Path
import argparse,csv,importlib.util,json,subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'frozen'))
from bioffn_v16 import confirmation_statistics as S
spec=importlib.util.spec_from_file_location('tahoe_frozen',ROOT/'statistics/core/tahoe_statistics.py');T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
METRICS=('mse','center_cosine_loss')
FAMILIES={'R': [('BioMeme','memory')], 'K':[('BioMeme','reference_informed_MLP'),('BioMeme','same_head_MLP')], 'P':[('memory','linear'),('memory','MLP')]}


def run_core(output):
    output.mkdir(parents=True,exist_ok=False);expected=json.loads((ROOT/'statistics/core/expected.json').read_text());rows=[];worst=0.
    for resource in ('U2OS','sciPlex3','Tahoe'):
        with np.load(ROOT/f'statistics/core/{resource}.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
        groups=data['groups'];indices=data['bootstrap_indices']
        np.testing.assert_array_equal(indices,np.random.default_rng(20261785).integers(0,len(groups),(10000,len(groups))))
        plan=S.BootstrapPlan(groups,indices,20261785)
        class Pop:
            def means(self,x):return np.concatenate([x[indices[i:i+500]].mean(axis=1) for i in range(0,10000,500)])
        for family,pairs in FAMILIES.items():
            tests=[]
            for left,right in pairs:
                for j,metric in enumerate(METRICS):
                    delta=data[left][:,j]-data[right][:,j]
                    result=(T.summarize(delta,Pop(),test=True) if resource=='Tahoe' else S.paired_summary(delta,groups,plan=plan))
                    tests.append(dict(result,left=left,right=right,metric=metric,first_mean=float(data[left][:,j].mean()),comparator_mean=float(data[right][:,j].mean())))
            if resource=='Tahoe':T.holm(tests)
            else:
                for test,adjusted in zip(tests,S.holm_adjust([x['p_value'] for x in tests])):
                    test['p_holm']=float(adjusted);test['conclusion']=('unresolved' if adjusted>.05 else 'method_better' if test['estimate']<0 else 'comparator_better')
            old=expected[resource][family]['tests'];assert len(old)==len(tests)
            for result,archive in zip(tests,old):
                assert result['metric']==archive['metric']
                for key in ('estimate','ci95','p_value','p_holm'):
                    a=np.asarray(result[key],float);b=np.asarray(archive[key],float);error=float(np.max(abs(a-b)));worst=max(worst,error);np.testing.assert_allclose(a,b,rtol=0,atol=1e-10,err_msg=f'{resource}/{family}/{key}')
                assert result['conclusion']==archive['conclusion']
                rows.append(dict(resource=resource,family=family,left=result['left'],right=result['right'],metric=result['metric'],first_mean=result['first_mean'],comparator_mean=result['comparator_mean'],estimate=result['estimate'],ci_lower=result['ci95'][0],ci_upper=result['ci95'][1],p_value=result['p_value'],p_holm=result['p_holm'],conclusion=result['conclusion'],n_compounds=len(groups)))
    directions=[r for r in rows if r['metric']=='center_cosine_loss' and (r['left'],r['right']) in [('BioMeme','same_head_MLP'),('memory','linear'),('memory','MLP')]]
    assert len(directions)==9 and all(r['conclusion']=='method_better' for r in directions)
    with (output/'table1_and_table2_key_rows.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    report=dict(status='passed',comparisons=30,original_direction_tests=9,direction_tests_favourable=9,maximum_absolute_difference_from_archive=worst,bootstrap_draws=10000,bootstrap_seed=20261785,bootstrap_matrices_identical=True,scopes=['Table 1: all five common loss contrasts for U2OS, sci-Plex3 and Tahoe; original 9/9 direction tests','Table 2: all-reference same-head and reference-informed MLP comparisons, with exact arm means'],pooling='Archived per-compound losses preserve seed-score averaging, within-compound task averaging and resource-specific target averaging; no prediction averaging introduced.',statistical_units='292 U2OS, 120 sci-Plex3, 277 Tahoe compounds')
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('stats_results'));args=parser.parse_args();start=time.perf_counter();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    core=run_core(out/'core')
    subprocess.run([sys.executable,str(ROOT/'statistics/a5/run_a5.py'),'--output',str(out/'a5')],check=True,env=dict(os.environ,PYTHONPATH=''))
    result=dict(status='passed',elapsed_seconds=time.perf_counter()-start,core=core,a5_validation='a5/REPLAY_VERIFICATION.json')
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
