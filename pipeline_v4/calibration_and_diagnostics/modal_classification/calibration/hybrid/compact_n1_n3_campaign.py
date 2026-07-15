"""Campaña compacta y sin fuga para N1/N3 sobre los 114 viajes ampliados."""
from __future__ import annotations

import hashlib, json, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT),str(HERE)]
from experimentar_n1_caminar import M2I, MODES, WALK, N1_PEDESTRIAN, make_features
from pipeline_v4.src import config
from pipeline_v4.src.random_forest_contract import RF_FEATURES, RF_HYPERPARAMETERS

OUT=ROOT/"Temporary outputs"/"ML compact N1 N3"
N3_SUBSET=[
 "drive_mean_speed","drive_max_speed","drive_std_speed","drive_stop_frac","drive_p25_speed","drive_p50_speed","drive_p75_speed",
 "drive_max_speed_diff","drive_mean_speed_diff","drive_highway_motorway_frac","drive_highway_residential_frac","drive_near_bus_frac",
 "drive_num_stops","drive_mean_stop_duration","drive_mean_stop_interval","drive_std_stop_interval","mean_snap_dist_drive","max_snap_dist_drive",
 "std_snap_dist_drive","drive_near_bus_drift_decay","drive_near_bus_high_drift","drive_win_near_bus_max","drive_win_near_bus_p90",
 "drive_win_near_bus_consec_run","drive_win_stops_max"]
N1_CONFIGS={
 "n1_rf52":{"kind":"rf","features":list(RF_FEATURES)},
 "n1_gb100_005_d2_52":{"kind":"gb","params":(100,.05,2),"features":list(RF_FEATURES)},
 "n1_gb100_005_d2_walk16":{"kind":"gb","params":(100,.05,2),"features":list(N1_PEDESTRIAN)},
 "n1_gb100_010_d2_walk16":{"kind":"gb","params":(100,.10,2),"features":list(N1_PEDESTRIAN)},
 "n1_gb150_005_d1_walk16":{"kind":"gb","params":(150,.05,1),"features":list(N1_PEDESTRIAN)},
}

class SoftBinaryEnsemble:
 def __init__(self,models): self.models=models; self.classes_=np.array([0,1])
 def predict_proba(self,X): return np.mean([m.predict_proba(X) for m in self.models],axis=0)
 def predict(self,X): return (self.predict_proba(X)[:,1]>=.5).astype(int)

def n1_model(cfg):
 if cfg["kind"]=="rf": return RandomForestClassifier(**RF_HYPERPARAMETERS["n1"])
 n,lr,d=cfg["params"]; return GradientBoostingClassifier(n_estimators=n,learning_rate=lr,max_depth=d,min_samples_leaf=4,random_state=42)

def n3_model(kind):
 if kind=="rf": return RandomForestClassifier(**RF_HYPERPARAMETERS["n3"])
 if kind=="gb": return GradientBoostingClassifier(n_estimators=100,learning_rate=.05,max_depth=2,min_samples_leaf=4,random_state=42)
 if kind=="hist": return HistGradientBoostingClassifier(max_iter=100,learning_rate=.05,max_depth=3,min_samples_leaf=4,l2_regularization=1.,random_state=42)
 return ExtraTreesClassifier(n_estimators=200,max_depth=8,min_samples_leaf=2,class_weight="balanced",random_state=42,n_jobs=3)

def fit_binary(model,X,y,weighted=False):
 if weighted and not isinstance(model,(RandomForestClassifier,ExtraTreesClassifier)):
  model.fit(X,y,sample_weight=compute_sample_weight("balanced",y))
 else: model.fit(X,y)
 return model

def splits_for(df,repeats):
 y=df.label.map(M2I); out=[]
 for rep,seed in enumerate([42,137][:repeats]):
  cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
  for fold,(tr,te) in enumerate(cv.split(df,y,df.caid_trip)): out.append((rep,fold,tr,te))
 return out

def fit_n2(X,y):
 mask=y!=WALK; m=RandomForestClassifier(**RF_HYPERPARAMETERS["n2"]); m.fit(X.loc[mask,list(RF_FEATURES)],(y[mask]==2).astype(int)); return m

def fit_n3(kind,X,y,features,ensemble_kinds=None):
 mask=y.isin([0,1]); target=(y[mask]==1).astype(int)
 if ensemble_kinds:
  return SoftBinaryEnsemble([fit_binary(n3_model(k),X.loc[mask,features],target,True) for k in ensemble_kinds])
 return fit_binary(n3_model(kind),X.loc[mask,features],target,True)

def predict_cascade(n1,n1f,n2,n3,n3f,threshold,X):
 pn1=n1.predict(X[n1f]); pred=np.full(len(X),WALK,dtype=int); motor=np.flatnonzero(pn1==1)
 if len(motor):
  metro=n2.predict(X.iloc[motor][list(RF_FEATURES)]).astype(bool); pred[motor[metro]]=2; surface=motor[~metro]
  if len(surface): pred[surface]=np.where(n3.predict_proba(X.iloc[surface][n3f])[:,1]>=threshold,1,0)
 return pred

def metric(y,p,fold_scores=None):
 r=recall_score(y,p,labels=range(4),average=None,zero_division=0); q=precision_score(y,p,labels=range(4),average=None,zero_division=0)
 return {"balanced_accuracy":balanced_accuracy_score(y,p),"macro_f1":f1_score(y,p,average="macro"),"recall_walk":r[3],"precision_walk":q[3],
  "recall_bus":r[1],"precision_bus":q[1],"recall_car":r[0],"recall_metro":r[2],"bus_to_car":int(np.sum((np.asarray(y)==1)&(np.asarray(p)==0))),
  "car_to_bus":int(np.sum((np.asarray(y)==0)&(np.asarray(p)==1))),"missed_walk":int(np.sum((np.asarray(y)==3)&(np.asarray(p)!=3))),
  "false_walk":int(np.sum((np.asarray(y)!=3)&(np.asarray(p)==3))),"fold_std":float(np.std(fold_scores)) if fold_scores is not None else np.nan}

def evaluate(df,splits,n1cfg,n3kind,n3features,threshold_mode=False,ensemble=None,name=""):
 X=df[list(RF_FEATURES)]; y=df.label.map(M2I).astype(int); records=[]; fold_rows=[]
 for rep,fold,tr,te in splits:
  assert set(df.iloc[tr].caid_trip).isdisjoint(set(df.iloc[te].caid_trip))
  cfg=N1_CONFIGS[n1cfg]; a=n1_model(cfg); t0=time.perf_counter(); fit_binary(a,X.iloc[tr][cfg["features"]],(y.iloc[tr]!=WALK).astype(int))
  b=fit_n2(X.iloc[tr],y.iloc[tr]); c=fit_n3(n3kind,X.iloc[tr],y.iloc[tr],n3features,ensemble)
  threshold=tune_threshold_n3(df.iloc[tr].reset_index(drop=True),n1cfg,n3kind,n3features,ensemble,700+rep*10+fold) if threshold_mode else .5
  train_s=time.perf_counter()-t0; t1=time.perf_counter(); p=predict_cascade(a,cfg["features"],b,c,n3features,threshold,X.iloc[te]); infer=time.perf_counter()-t1
  yt=y.iloc[te].values; fold_rows.append({"pipeline":name,"repeat":rep,"fold":fold,"threshold":threshold,"train_seconds":train_s,"inference_seconds":infer,"n_test":len(te),**metric(yt,p)})
  records += [{"pipeline":name,"repeat":rep,"fold":fold,"row":int(i),"caid_trip":df.iloc[i].caid_trip,"deg":df.iloc[i].deg,"label":MODES[y.iloc[i]],"prediction":MODES[v]} for i,v in zip(te,p)]
 return pd.DataFrame(records),pd.DataFrame(fold_rows)

def tune_threshold_n3(train,n1cfg,n3kind,n3features,ensemble,seed):
 X=train[list(RF_FEATURES)]; y=train.label.map(M2I).astype(int); probs=[]; truth=[]
 cv=StratifiedGroupKFold(3,shuffle=True,random_state=seed)
 # Umbral se estima sólo sobre casos viales reales del train; no consulta el test exterior.
 road=y.isin([0,1]); road_df=train[road].reset_index(drop=True); Xr=road_df[list(RF_FEATURES)]; yr=road_df.label.map(M2I).astype(int)
 for tr,va in cv.split(Xr,yr,Xr.index.map(lambda i:road_df.iloc[i].caid_trip)):
  m=fit_n3(n3kind,Xr.iloc[tr],yr.iloc[tr],n3features,ensemble); probs.extend(m.predict_proba(Xr.iloc[va][n3features])[:,1]); truth.extend((yr.iloc[va]==1).astype(int))
 scores=[(balanced_accuracy_score(truth,np.asarray(probs)>=t),t) for t in [.40,.45,.50,.55]]
 return max(scores,key=lambda x:(x[0],-abs(x[1]-.5)))[1]

def summarize(oof,folds):
 name_to_idx={m:i for i,m in enumerate(MODES)}; y=oof.label.map(name_to_idx).values; p=oof.prediction.map(name_to_idx).values; result=metric(y,p,folds.balanced_accuracy.values)
 result["train_seconds"]=folds.train_seconds.sum(); result["inference_ms_per_scenario"]=1000*folds.inference_seconds.sum()/folds.n_test.sum(); return result

def main():
 OUT.mkdir(parents=True,exist_ok=True); df=make_features("datos_entrenamiento_ml_expanded.pkl"); screen=splits_for(df,1); rows=[]
 # A: N1 rápido con N3 RF actual.
 for n1 in N1_CONFIGS:
  name=f"screen_{n1}__n3_rf52"; o,f=evaluate(df,screen,n1,"rf",list(RF_FEATURES),name=name); rows.append({"stage":"n1_screen","pipeline":name,**summarize(o,f)})
 n1best=max(rows,key=lambda r:(r["balanced_accuracy"],r["recall_walk"]))["pipeline"].split("__")[0].replace("screen_","")
 # B: familias N3 x dos contratos.
 screened=[]
 for kind in ["rf","gb","hist","extra"]:
  for tag,features in [("52",list(RF_FEATURES)),("25",N3_SUBSET)]:
   name=f"screen_{n1best}__n3_{kind}{tag}"; o,f=evaluate(df,screen,n1best,kind,features,name=name); s={"stage":"n3_screen","pipeline":name,**summarize(o,f)}; rows.append(s); screened.append((s,kind,features))
 ranked=sorted(screened,key=lambda z:(z[0]["balanced_accuracy"],z[0]["recall_bus"],z[0]["recall_walk"]),reverse=True)
 # Ensamble de las dos mejores familias, usando el contrato del mejor.
 topk=[]
 for _,k,_ in ranked:
  if k not in topk: topk.append(k)
  if len(topk)==2: break
 ens_features=ranked[0][2]; ens_name=f"screen_{n1best}__n3_ensemble_{topk[0]}_{topk[1]}_{len(ens_features)}"
 o,f=evaluate(df,screen,n1best,"ensemble",ens_features,ensemble=topk,name=ens_name); ens_s={"stage":"n3_screen","pipeline":ens_name,**summarize(o,f)}; rows.append(ens_s); screened.append((ens_s,"ensemble",ens_features,topk))
 # C: umbral sólo para dos mejores N3 y después validación repetida de los dos pipelines completos.
 top2=sorted(screened,key=lambda z:(z[0]["balanced_accuracy"],z[0]["recall_bus"]),reverse=True)[:2]; finalists=[]
 repeated=splits_for(df,2)
 for item in top2:
  s,kind,features,*rest=item; ensemble=rest[0] if rest else None; name=f"final_{n1best}__{kind}_{len(features)}_nested_threshold"
  o,f=evaluate(df,repeated,n1best,kind,features,True,ensemble,name); summary={"stage":"repeated10","pipeline":name,**summarize(o,f)}; rows.append(summary); finalists.append((summary,o,f,kind,features,ensemble))
 best=max(finalists,key=lambda z:(z[0]["balanced_accuracy"],z[0]["recall_bus"],z[0]["recall_walk"],z[0]["recall_car"],-z[0]["fold_std"]))
 summary,oof,folds,kind,features,ensemble=best; pd.DataFrame(rows).to_csv(OUT/"compact_n1_n3_experiments.csv",index=False); oof.to_csv(OUT/"predicciones_oof_mejor_candidato.csv",index=False)
 name_to_idx={m:i for i,m in enumerate(MODES)}; cm=confusion_matrix(oof.label.map(name_to_idx),oof.prediction.map(name_to_idx),labels=range(4)); np.savetxt(OUT/"matriz_confusion_mejor_candidato.csv",cm,fmt="%d",delimiter=",")
 # Fit final y umbral train-only OOF.
 X=df[list(RF_FEATURES)]; y=df.label.map(M2I).astype(int); cfg=N1_CONFIGS[n1best]; a=fit_binary(n1_model(cfg),X[cfg["features"]],(y!=WALK).astype(int)); b=fit_n2(X,y); c=fit_n3(kind,X,y,features,ensemble)
 threshold=tune_threshold_n3(df,n1best,kind,features,ensemble,999)
 contract={"version":"ML_expanded_compact_candidate","n1":{"model":type(a).__name__,"features":cfg["features"]},"n2":{"model":type(b).__name__,"features":list(RF_FEATURES)},
  "n3":{"model":type(c).__name__,"base_models":ensemble,"features":features,"threshold_bus":threshold}}
 artifact={"clf_n1":a,"clf_n2":b,"clf_n3":c,"feature_cols_v4":list(RF_FEATURES),"feature_cols_new":list(RF_FEATURES),"model_contract":contract,
  "metadata":{"physical_trips":114,"scenarios":445,"validation":summary,"dataset":"datos_entrenamiento_ml_expanded.pkl"}}
 model_path=OUT/"modal_classifier_compact_n1_n3_candidate.pkl"; pickle.dump(artifact,open(model_path,"wb"))
 (OUT/"modelos_variables_umbrales.json").write_text(json.dumps(contract,indent=2,ensure_ascii=False),encoding="utf-8")
 # Comparación explícita con métricas robustas previas de producción.
 prod={"balanced_accuracy":.8502604,"recall_walk":.875,"recall_bus":.6510417,"recall_car":.896875,"recall_metro":1.0}
 desirable=(summary["balanced_accuracy"]>=.84 and summary["recall_bus"]>=.65 and summary["recall_walk"]>=.75 and summary["recall_car"]>=.93 and summary["recall_metro"]>=.98)
 report=["# Campaña compacta N1 + N3","",f"Mejor N1: **{n1best}**. Mejor pipeline repetido: **{summary['pipeline']}**.","",
  pd.DataFrame(rows).to_markdown(index=False,floatfmt=".4f"),"","## Matriz acumulada del mejor candidato","","```text",str(cm),"```","",
  "## Decisión","",f"Objetivos deseables completos: **{desirable}**. Producción robusta previa: {prod}.",
  ("**Recomendación: promover tras una revisión manual final; el artefacto quedó separado y producción no fue modificada.**" if desirable and summary["balanced_accuracy"]>=prod["balanced_accuracy"]-.01 else "**Recomendación: mantener producción. El candidato ampliado queda separado para evaluación posterior.**"),"",
  "No se modificaron dataset, ruteo, variables, N2, orquestador ni modelo desplegado. El umbral Bus se eligió exclusivamente mediante OOF agrupado del train."]
 (OUT/"compact_n1_n3_report.md").write_text("\n".join(report),encoding="utf-8")
 print(json.dumps({"n1_best":n1best,"best":summary,"threshold":float(threshold),"desirable":bool(desirable),"model":str(model_path)},indent=2))

if __name__=="__main__": main()

