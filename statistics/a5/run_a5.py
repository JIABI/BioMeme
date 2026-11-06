#!/usr/bin/env python3
"""Portable replay of the archived independent A5 calculation.
Only paths, output handling and final archive checks were adapted in revision 4.9.
This replay is not a new independent or blinded scientific verification.
"""
from pathlib import Path
import csv,hashlib,json,shutil,datetime,collections
import numpy as np
import argparse
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description='Replay A5 from stored weights, annotations and sampled identities.')
parser.add_argument('--output',type=Path,default=HERE/'replayed')
args=parser.parse_args()
OUT=args.output.resolve(); OUT.mkdir(parents=True,exist_ok=True)
BASE=HERE/'inputs'
PLAN=BASE/'OPAL2_biology_kernel_BioMeme_A5_mechanism_weights_prespec_20261102.md'
SOURCES=['smoother/per_query_compound_weights.csv','moa/annotations.json','moa/candidate_pools.json','moa/observed_neighbours.csv','moa/observed_neighbour_scores.npz','moa/random_neighbours.npz','moa/random_distribution.csv']
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def savej(name,obj):
 (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def savecsv(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
shutil.copyfile(PLAN,OUT/PLAN.name)
savej('INPUT_MANIFEST.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan':{'path':str(PLAN),'sha256':digest(PLAN)},'inputs':[{'relative_path':f,'source_path':str(BASE/f),'bytes':(BASE/f).stat().st_size,'sha256':digest(BASE/f)} for f in SOURCES]})
annotations=json.loads((BASE/'moa/annotations.json').read_text())
labels={k:frozenset(map(tuple,v['typed_labels'])) for k,v in annotations.items()}
pools=json.loads((BASE/'moa/candidate_pools.json').read_text())
q_by_row={qid:p for p in pools for qid in p['query_ids']}
compounds=sorted(p['query_compound'] for p in pools)
assert len(compounds)==len(set(compounds))==224
assert len(q_by_row)==896 and all(len(p['query_ids'])==4 for p in pools)
bycompound={p['query_compound']:p for p in pools}
assert compounds==[p['query_compound'] for p in pools]
def shared(q,c):return bool(labels[q]&labels[c])
# Reconstruct registered similarity-neighbour fractions from identities.
observed=collections.defaultdict(list)
with (BASE/'moa/observed_neighbours.csv').open() as f:
 for r in csv.DictReader(f):
  if r['arm']!='memory':continue
  p=q_by_row[r['query_id']];q=p['query_compound'];c=r['neighbour_compound']
  assert q==r['query_compound'] and p['episode']==r['episode']
  assert c!=q and c in p['candidate_compounds']
  m=int(shared(q,c));assert m==int(r['shared_annotation'])
  observed[r['query_id']].append((int(r['rank']),c,m))
assert set(observed)==set(q_by_row)
simrow={}
for qid,v in observed.items():
 assert len(v)==5 and len(set(c for _,c,_ in v))==5
 assert sorted(rank for rank,_,_ in v)==[1,2,3,4,5]
 simrow[qid]=sum(m for _,_,m in v)/5
simcompound=np.array([np.mean([simrow[qid] for qid in bycompound[q]['query_ids']]) for q in compounds])
a=np.load(BASE/'moa/observed_neighbour_scores.npz',allow_pickle=False)
assert list(a['compound_ids'])==compounds
assert np.max(np.abs(simcompound-a['compound_fraction__memory']))<1e-12
assert abs(float(simcompound.mean())-0.1015625)<1e-12
# Reconstruct existing random draws directly from saved sampled identities.
b=np.load(BASE/'moa/random_neighbours.npz',allow_pickle=False)
assert list(b['query_compounds'])==compounds
candidate_dict=b['candidate_compounds'].tolist();idx=b['selected_candidate_indices'];assert idx.shape==(1000,224,5)
matched=np.array([[shared(q,c) for c in candidate_dict] for q in compounds],dtype=bool)
valid=np.array([[c in bycompound[q]['candidate_compounds'] and c!=q for c in candidate_dict] for q in compounds],dtype=bool)
for i,q in enumerate(compounds):
 assert valid[i,idx[:,i,:]].all()
 assert np.all(np.diff(np.sort(idx[:,i,:],axis=1),axis=1)>0)
random_compound=matched[np.arange(224)[None,:,None],idx].mean(axis=2)
random_pooled=random_compound.mean(axis=1)
assert np.max(np.abs(random_compound-b['random_fraction_by_compound']))<1e-12
assert np.max(np.abs(random_pooled-b['pooled_random_fraction']))<1e-12
random_mean=float(random_pooled.mean());random_ci=np.percentile(random_pooled,[2.5,97.5])
assert abs(random_mean-0.03417589285714286)<1e-12
assert np.max(np.abs(random_ci-[0.025,0.04464285714285714]))<1e-12
gate={'pass':True,'n_compounds':224,'n_query_rows':896,'similarity_fraction':float(simcompound.mean()),'random_mean':random_mean,'random_central_95':random_ci.tolist(),'random_draws':1000,'all_original_sampled_identities_valid':True}
savej('GATES_PASSED_BEFORE_A5_RANKING.json',gate)
print('Original-result gates passed.',flush=True)
# Index existing complete scalar weights; retain only H4 annotated query rows.
weights=collections.defaultdict(dict);totalquery=set()
with (BASE/'smoother/per_query_compound_weights.csv').open() as f:
 for r in csv.DictReader(f):
  qid=r['query_id'];totalquery.add(qid)
  if qid not in q_by_row:continue
  p=q_by_row[qid];assert r['query_compound']=='LKCP_GROUP::'+p['query_compound']
  assert r['context']+'/'+r['episode']==p['episode']
  assert r['memory_compound'].startswith('LKCP_GROUP::')
  c=r['memory_compound'].split('::',1)[1]
  assert c not in weights[qid]
  weights[qid][c]=r
assert len(totalquery)==1168 and set(weights)==set(q_by_row)
rowout=[];candidateout=[];selectedout=[]
for q in compounds:
 p=bycompound[q]
 for qid in sorted(p['query_ids']):
  wd=weights[qid]
  primary=set(p['candidate_compounds'])-{q}
  secondary={c for c in wd if c!=q and labels.get(c)}
  assert primary<=secondary and all(c in wd for c in primary)
  assert len(primary)>=5 and len(secondary)>=5
  rank=lambda cc:sorted(cc,key=lambda c:(-float(wd[c]['complete_weight']),c))
  rp,rs=rank(primary),rank(secondary);rpi={c:i+1 for i,c in enumerate(rp)};rsi={c:i+1 for i,c in enumerate(rs)}
  valp=np.mean([shared(q,c) for c in rp[:5]]);vals=np.mean([shared(q,c) for c in rs[:5]])
  randoms=np.mean([shared(q,c) for c in secondary]);randomp=np.mean([shared(q,c) for c in primary])
  rowout.append(dict(query_id=qid,query_compound=q,episode=p['episode'],n_primary_candidates=len(primary),n_secondary_candidates=len(secondary),primary_fraction=float(valp),secondary_fraction=float(vals),similarity_fraction=simrow[qid],primary_paired_difference=float(valp-simrow[qid]),primary_random_exact=float(randomp),secondary_random_exact=float(randoms)))
  for c in sorted(secondary):
   common=sorted(labels[q]&labels[c]);wr=wd[c]
   candidateout.append(dict(query_id=qid,query_compound=q,episode=p['episode'],candidate_compound=c,candidate_name=annotations[c]['name'],memory_role=wr['memory_role'],complete_weight=float(wr['complete_weight']),in_primary=int(c in primary),rank_primary=rpi.get(c,''),rank_secondary=rsi[c],selected_primary=int(c in rp[:5]),selected_secondary=int(c in rs[:5]),shared_annotation=int(bool(common)),shared_typed_labels=json.dumps(common,ensure_ascii=False)))
  for scope,ranking in [('primary',rp),('secondary',rs)]:
   for i,c in enumerate(ranking[:5],1):selectedout.append(dict(scope=scope,query_id=qid,query_compound=q,episode=p['episode'],rank=i,candidate_compound=c,candidate_name=annotations[c]['name'],complete_weight=float(wd[c]['complete_weight']),shared_annotation=int(shared(q,c))))
rowmap={r['query_id']:r for r in rowout};compoundout=[]
for q in compounds:
 rlist=[rowmap[i] for i in bycompound[q]['query_ids']]
 cr={'query_compound':q,'query_name':bycompound[q]['query_name'],'n_query_rows':len(rlist)}
 for field in ['primary_fraction','secondary_fraction','similarity_fraction','primary_paired_difference','primary_random_exact','secondary_random_exact']:cr[field]=float(np.mean([r[field] for r in rlist]))
 compoundout.append(cr)
primary=np.array([r['primary_fraction'] for r in compoundout]);secondary=np.array([r['secondary_fraction'] for r in compoundout]);paired=primary-simcompound
# One matrix, whole compounds, sorted identities, default percentile interpolation.
boot_idx=np.random.default_rng(20261102).integers(0,224,size=(10000,224))
bp=primary[boot_idx].mean(axis=1);bd=paired[boot_idx].mean(axis=1);bs=secondary[boot_idx].mean(axis=1)
def estimate(v,draws):return {'estimate':float(v.mean()),'ci95':np.percentile(draws,[2.5,97.5]).tolist()}
secondary_random=float(np.mean([r['secondary_random_exact'] for r in compoundout]))
res={'schema':'A5_GPT_independent_v1','population':{'compounds':224,'query_rows':896,'all_query_rows_in_weight_export':1168},'gates':gate,'bootstrap':{'seed':20261102,'draws':10000,'sampling_unit':'whole compound','compound_order':'ascending exact BRD ID','percentile_method':'linear'},'primary':estimate(primary,bp),'primary_minus_similarity':estimate(paired,bd),'similarity':float(simcompound.mean()),'random_baseline':{'mean':random_mean,'central95':random_ci.tolist(),'primary_analytic_mean':float(np.mean([r['primary_random_exact'] for r in compoundout]))},'primary_to_random_ratio':float(primary.mean()/random_mean),'secondary':estimate(secondary,bs),'secondary_random_analytic_mean':secondary_random,'secondary_to_random_ratio':float(secondary.mean()/secondary_random)}
res['reporting_rule']={'primary_threshold':0.0446,'primary_exceeds_random_threshold':bool(primary.mean()>0.0446),'paired_interval_above_zero':bool(np.percentile(bd,2.5)>0),'primary_location':'main text' if primary.mean()>0.0446 else 'SI with main-text sentence','secondary_location':'SI only','paired_claim_allowed':bool(primary.mean()>0.0446 and np.percentile(bd,2.5)>0)}
savecsv('query_candidate_weights_and_matches.csv',candidateout);savecsv('selected_five_by_query.csv',selectedout);savecsv('query_outcomes.csv',rowout);savecsv('compound_outcomes.csv',compoundout)
np.savez_compressed(OUT/'bootstrap_draws.npz',compound_ids=np.array(compounds),sampled_compound_indices=boot_idx,primary_draws=bp,paired_difference_draws=bd,secondary_draws=bs)
savej('INDEPENDENT_RESULTS.json',res)
files=[p for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='INDEPENDENT_OUTPUT_MANIFEST.json']
savej('INDEPENDENT_OUTPUT_MANIFEST.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results_saved_before_external_comparison':True,'files':[{'filename':p.name,'sha256':digest(p),'bytes':p.stat().st_size} for p in files]})
print(json.dumps(res,indent=2),flush=True)

# Check the recomputed statistics only after saving the new calculation.
expected=json.loads((HERE/'expected_results.json').read_text())
max_error=[0.0]
def compare(actual, archived):
 if isinstance(archived,dict):
  assert set(actual)==set(archived)
  for key in archived: compare(actual[key],archived[key])
 elif isinstance(archived,list):
  assert len(actual)==len(archived)
  for x,y in zip(actual,archived):compare(x,y)
 elif isinstance(archived,(int,float)) and not isinstance(archived,bool):
  err=abs(actual-archived);max_error[0]=max(max_error[0],err);assert err<=1e-12
 else: assert actual==archived
compare(res,expected)
with np.load(HERE/'expected_bootstrap.npz',allow_pickle=False) as expected_draws:
 assert np.array_equal(boot_idx,expected_draws['sampled_compound_indices'])
 assert np.array_equal(np.array(compounds),expected_draws['compound_ids'])
 for value,name in [(bp,'primary_draws'),(bd,'paired_difference_draws'),(bs,'secondary_draws')]:
  assert np.max(np.abs(value-expected_draws[name]))<=1e-12
savej('REPLAY_VERIFICATION.json',{'passed':True,'max_statistic_absolute_error':max_error[0],
 'bootstrap_indices_identical':True,'compounds':224,'query_rows':896,
 'scope':'Stored complete weights and typed annotations through rankings, whole-compound statistics and archived reporting rule.'})
print('A5 replay PASS; maximum statistic absolute error =',max_error[0])
