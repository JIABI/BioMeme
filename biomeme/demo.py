"""Array I/O and orchestration around verbatim frozen scientific functions.

The query outcome file is opened only after prediction.npz is sealed.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, csv, gzip, hashlib, json, os, platform, resource, subprocess, sys, time
import numpy as np
import scipy
from bioffn_v11.multiwell import MultiwellCalibration, crossfit_multiwell, build_streams
from bioffn_v8.memory import kernel_geometry
from bioffn_v16.memory import fit_memory, kernel_matrix
from bioffn_v16.crosscell_oof import generate_oof_errors
from bioffn_v16.biological_uncertainty import posterior_variance
from bioffn_v16.gated_transfer import gate_x
from bioffn_v16.transfer import trajectory_rows
from bioffn_v7.baselines import group_weights
from bioffn_v5.metrics import cosine_loss
from biomeme.confirmation_adapter import fit_k4c_head, donor_arrays
from biomeme.validation import compare_source_residuals


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=lambda v:v.tolist() if hasattr(v,"tolist") else str(v))+"\n")
def loadnp(path):
    with np.load(path,allow_pickle=False) as f: return {k:f[k] for k in f.files}
def dataset(data,stem):
    a=loadnp(data/(stem+".npz")); a["metadata"]=json.loads((data/(stem+".json")).read_text())
    return SimpleNamespace(**a)
def subset(data,ix):
    ix=np.asarray(ix,int);out={k: v[ix] for k,v in vars(data).items() if isinstance(v,np.ndarray)}
    out["metadata"]=dict(data.metadata,row_inventory=[data.metadata["row_inventory"][i] for i in ix])
    return SimpleNamespace(**out)
def codes(data): return np.asarray([f"CP{r['target_time']}|task{r['task']}" for r in data.metadata["row_inventory"]])
def numeric_conditions(data): return np.asarray([(r['target_time'],r['task']) for r in data.metadata['row_inventory']])
def streams(data):
    value=build_streams(data,"available_time")
    return value["streams"],value["active"]
def rel(a,b): return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-15))
def guard():
    if sys.platform=="darwin":
        r=subprocess.run(["sysctl","-n","kern.memorystatus_vm_pressure_level"],capture_output=True,text=True)
        if r.returncode==0 and int(r.stdout.strip())>=4: raise RuntimeError("Critical system memory pressure; stop and rerun when free memory is available.")
def fit_G(data):
    x,a=streams(data);return MultiwellCalibration().fit(x,a,data.target,codes(data),data.groups)
def predict_G(model,data):
    x,a=streams(data);return model.predict(x,a,codes(data))
def source_oof(data,seed):
    x,a=streams(data);return crossfit_multiwell(x,a,data.target,codes(data),data.groups,n_folds=8,seed=seed)[0]


def run_fold(root,out,fold):
    start=time.perf_counter();guard();data=root/"data/u2os_demo";out.mkdir(parents=True,exist_ok=False)
    events=[]
    def event(value):
        record=dict(seconds=time.perf_counter()-start,**value);events.append(record)
        with (out/"progress.jsonl").open("a") as f:f.write(json.dumps(record,default=str)+"\n")
    core=dataset(data,"core");ref=dataset(data,f"fold{fold}_references");query=dataset(data,f"fold{fold}_query")
    assert not hasattr(query,"target")
    assert set(core.groups).isdisjoint(query.groups) and set(ref.groups).isdisjoint(query.groups)
    bank=loadnp(data/"coordinate_bank.npz")["rows"]
    index=json.loads((data/"coordinate_indices.json").read_text())[str(fold)]
    z={k:bank[ix].transpose(1,0,2) for k,ix in index.items()}
    del bank
    foldmeta=json.loads(gzip.decompress((data/f"fold{fold}_folds.json.gz").read_bytes()))
    G=fit_G(core);rs=core.target.astype(float)-source_oof(core,20261008+200009)
    rr=ref.target.astype(float)-predict_G(G,ref);gq=predict_G(G,query)
    R=np.concatenate((rs,rr));ids=np.r_[core.ids,ref.ids];groups=np.r_[core.groups,ref.groups]
    models=[];memory=np.zeros_like(gq);weights=[];variances=[];member_checks=[]
    for k in range(8):
        guard();X=z["final_z"][k];Q=z["query_z"][k];geo=kernel_geometry(X[:len(core.ids)])
        m=fit_memory(X[:len(core.ids)],rs,core.groups,geometry=geo,reference_features=X[len(core.ids):],reference_residual=rr,reference_groups=ref.groups,variant="main",source_codes=codes(core),reference_codes=codes(ref))
        value=m.predict(Q,query_codes=codes(query));memory+=value/8;models.append(m)
        # Independently expressed fixed smoother, including its unpenalized mean.
        gram=kernel_matrix(X,X,geo);K=kernel_matrix(X,Q,geo);v=m.v15_model
        g=group_weights(groups);w=len(g)*g;h=v.train_h;q=v.q;phi=np.ones(len(X));hq=np.ones(len(Q))/np.sqrt(q)
        L=gram*h[:,None]*h[None,:];L=(L+L.T)/2;L.flat[::len(L)+1]+=100./w
        s=np.linalg.solve(L,(hq[:,None]*K*h[None,:]).T).T
        c=s+(1.-s@phi)[:,None]*(g*phi/q)[None,:]
        check=dict(provider=k,weight_sum_max_abs_error=float(np.max(abs(c.sum(1)-1))),residual_prediction_relative_error=rel(c@R,value))
        assert check['weight_sum_max_abs_error']<=1e-10 and check['residual_prediction_relative_error']<=1e-10
        variance,vaudit=posterior_variance(gram,K,np.diag(kernel_matrix(Q,Q,geo)),row_weights=w,train_h=h,query_h=hq,ridge=100.)
        variances.append(variance);check['latent_krr_variance']=vaudit
        member_checks.append(check);weights.append(c)
    event(dict(stage="final_G_and_memory_fitted",source_rows=len(core.ids),reference_rows=len(ref.ids)))
    packs=[]
    for meta in foldmeta:
        guard();pack=dict(meta);a=meta['audit'];f=int(meta['fold'])
        ct=subset(core,a['source_train_indices']);cv=subset(core,a['source_valid_indices']);rt=subset(ref,a['reference_train_indices']);rv=subset(ref,a['reference_valid_indices'])
        local=fit_G(ct)
        pack['R']=np.concatenate((ct.target.astype(float)-source_oof(ct,20261008+f),rt.target.astype(float)-predict_G(local,rt)))
        pack['R_valid']=np.concatenate((cv.target.astype(float)-predict_G(local,cv),rv.target.astype(float)-predict_G(local,rv)))
        for field in ('ids','groups','ids_valid','groups_valid','valid_groups','phi','phi_valid'):pack[field]=np.asarray(pack[field])
        np.testing.assert_array_equal(pack['ids'],np.r_[ct.ids,rt.ids]);np.testing.assert_array_equal(pack['ids_valid'],np.r_[cv.ids,rv.ids])
        pack['z']=z[f'f{f}_z'];pack['z_valid']=z[f'f{f}_z_valid'];packs.append(pack)
    event(dict(stage="nested_G_residuals_refitted",inner_folds=8))
    oof=generate_oof_errors(packs,'memory',ids=ids,groups=groups,query_groups=query.groups,callback=event)
    oof['audit']['own_base_family']='memory'
    donor=loadnp(data/'donor.npz');donor['variant']='true'
    episode=dict(ids=ids,groups=groups,conditions=np.r_[numeric_conditions(core),numeric_conditions(ref)],query_ids=query.ids,query_groups=query.groups,query_conditions=numeric_conditions(query),base=np.zeros_like(R))
    head=fit_k4c_head(episode,dict(prediction=gq+memory,oof=oof,models=models,family='memory'),donor,callback=event)
    (_, _),(dq,tq)=donor_arrays(episode,donor);hm=head['model']
    direct=hm.beta*gate_x(dq,hm.c);lowrank=hm.basis.project(dq,tq)[:,:len(hm.coefficients)]@hm.coefficients
    terms=np.stack((gq,memory,direct,lowrank))
    decomposition=rel(terms.sum(0),head['prediction']);assert decomposition<=1e-10
    prediction_path=out/'prediction.npz'
    np.savez_compressed(prediction_path,ids=query.ids,groups=query.groups,prediction=head['prediction'],backbone=gq,memory_prediction=gq+memory,four_terms=terms)
    seal=dict(path='prediction.npz',sha256=sha(prediction_path),seconds=time.perf_counter()-start,query_outcomes_loaded=False)
    dump(out/'prediction_seal.json',seal)
    np.savez_compressed(out/'memory_weights.npz',query_ids=query.ids,memory_ids=ids,memory_groups=groups,mean_weights=np.mean(weights,axis=0),training_residuals=R,member_latent_krr_variance=np.asarray(variances),mean_latent_krr_variance=np.mean(variances,axis=0))
    dump(out/'selection.json',dict(selected_c=hm.c,selected_rank=len(hm.coefficients),selected_head=hm.audit,candidate_heads={str(c):m.audit for c,m in head['c_candidates'].items()},oof_audit=oof['audit']))
    # Scoring-only entry point: first read of withheld outcomes/archived predictions.
    score=loadnp(data/f'fold{fold}_scoring_only.npz');np.testing.assert_array_equal(score['ids'],query.ids)
    manifest=json.loads((data/'MANIFEST.json').read_text());expected=manifest['outer_folds'][fold]
    archived_residuals=loadnp(data/'archived_source_residuals.npz')
    np.testing.assert_array_equal(core.ids,archived_residuals['ids'])
    source_residual_check=compare_source_residuals(rs,archived_residuals['residuals'])
    archive_checks={name:rel(new,score[key]) for name,new,key in [('BioMeme',head['prediction'],'prediction'),('memory',gq+memory,'memory_prediction'),('backbone',gq,'backbone_prediction')]}
    assert max(archive_checks.values())<=1e-8,archive_checks
    assert hm.c==expected['selected_c'] and len(hm.coefficients)==expected['selected_rank']
    assert hm.audit['lambda_value']==expected['selected_head']['lambda_value']
    center=score['source_fit_centers'];axis=head['prediction']-center
    centered=terms.copy();centered[0]-=center
    shares=np.sum(centered*axis[None,:,:],axis=2)/np.maximum(np.sum(axis*axis,axis=1),1e-30)[None,:]
    assert np.max(abs(shares.sum(0)-1))<=1e-10
    np.savez_compressed(out/'decomposition.npz',ids=query.ids,term_names=np.asarray(['calibration','memory','conserved','low_rank']),four_terms=terms,projection_shares=shares,source_fit_centers=center)
    rows=[]
    for arm,pp in [('BioMeme',head['prediction']),('memory',gq+memory),('backbone',gq)]:
        mse=np.mean((pp-score['target'])**2,axis=1);direction=cosine_loss(pp-center,score['target']-center)
        for group in np.unique(query.groups):
            ix=query.groups==group;rows.append(dict(compound=group,outer_fold=fold,arm=arm,mse=float(mse[ix].mean()),direction_error=float(direction[ix].mean()),n_rows=int(ix.sum())))
    with (out/'compound_losses.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    report=dict(status='passed',outer_fold=fold,query_rows=len(query.ids),query_compounds=len(set(query.groups)),source_compounds=len(set(core.groups)),reference_compounds=len(set(ref.groups)),elapsed_seconds=time.perf_counter()-start,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),archive_prediction_relative_errors=archive_checks,selected_c=hm.c,selected_rank=len(hm.coefficients),selected_lambda=hm.audit['lambda_value'],decomposition_relative_error=decomposition,projection_share_sum_max_error=float(np.max(abs(shares.sum(0)-1))),memory_members=member_checks,prediction_sealed_before_scoring=True,query_outcomes_used_for_fit=False,source_residual_sha256=hashlib.sha256(rs.tobytes()).hexdigest(),archived_source_residual_sha256=expected['source_residual_sha256'])
    report['source_residual_numeric_check']=source_residual_check
    report['scientific_module_paths']={name:str(Path(mod.__file__).resolve().relative_to(root)) for name,mod in sys.modules.items() if name.startswith(('bioffn_', 'biomeme')) and getattr(mod,'__file__',None)}
    dump(out/'validation.json',report);print(json.dumps({k:report[k] for k in ('status','outer_fold','elapsed_seconds','archive_prediction_relative_errors','selected_c','selected_rank')}),flush=True)


def aggregate(root,out,elapsed):
    reports=[json.loads((out/f'fold{f}/validation.json').read_text()) for f in range(4)]
    rows=[]
    for f in range(4):
        with (out/f'fold{f}/compound_losses.csv').open() as handle: rows.extend(csv.DictReader(handle))
    with (out/'compound_losses.csv').open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    means={a:{metric:float(np.mean([float(r[metric]) for r in rows if r['arm']==a])) for metric in ('mse','direction_error')} for a in ('BioMeme','memory','backbone')}
    bio={r['compound']:r for r in rows if r['arm']=='BioMeme'};mem={r['compound']:r for r in rows if r['arm']=='memory'}
    assert len(bio)==101
    rng=np.random.default_rng(20261103);keys=sorted(bio);bootstrap={}
    for metric in ('mse','direction_error'):
        delta=np.asarray([float(bio[k][metric])-float(mem[k][metric]) for k in keys]);values=np.mean(delta[rng.integers(0,len(delta),size=(10000,len(delta)))],axis=1)
        bootstrap[metric]=dict(contrast='BioMeme minus memory',point=float(delta.mean()),percentile_95_interval=np.quantile(values,[.025,.975]).tolist(),resamples=10000,seed=20261103,unit='compound')
    result=dict(status='passed',context='U2OS__ASG003_to_LKCP001',budget='all',outer_folds=4,query_compounds=101,query_rows=404,output_coordinates=1694,provider_members=8,elapsed_seconds=elapsed,max_rss_bytes=max(r['max_rss_bytes'] for r in reports),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),means=means,paired_bootstrap=bootstrap,folds=reports,all_predictions_reproduced_with_relative_tolerance=1e-8,model_fits_performed=True,original_archive_required=False,entry_boundary='real processed arrays and frozen source-only fold-local 39-coordinate keys',provider='eight no-intercept ridge response providers, penalty 10000, fixed from independent Pilot1; provider inference exported as processed keys')
    dump(out/'validation.json',result);print(json.dumps(dict(status='passed',elapsed_seconds=elapsed,means=means)),flush=True)


def main(root):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('results'));parser.add_argument('--fold',type=int,choices=range(4),help=argparse.SUPPRESS);args=parser.parse_args();out=args.output.resolve()
    if args.fold is not None:return run_fold(root,out,args.fold)
    start=time.perf_counter();out.mkdir(parents=True,exist_ok=False)
    for fold in range(4):subprocess.run([sys.executable,str(root/'run_demo.py'),'--output',str(out/f'fold{fold}'),'--fold',str(fold)],check=True,env=dict(os.environ,PYTHONPATH=''))
    aggregate(root,out,time.perf_counter()-start)
