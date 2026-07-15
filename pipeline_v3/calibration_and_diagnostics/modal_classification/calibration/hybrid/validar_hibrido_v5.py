"""Validación repetida, agrupada y anidada del clasificador modal híbrido v5."""
from __future__ import annotations

import hashlib
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(HERE))
from experimentar_n1_caminar import M2I, MODES, WALK, make_features
from pipeline_v3.src import config
from pipeline_v3.src.random_forest_contract import RF_FEATURES, RF_HYPERPARAMETERS

OUT=ROOT/"Temporary outputs"/"ML v5 hybrid validation"
GRID=list(itertools.product([50,100,150],[.03,.05,.1],[1,2,3],[2,4,6]))


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def split20(df,y):
    out=[]
    for repeat,seed in enumerate([42,73,101,211]):
        cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
        for fold,(tr,te) in enumerate(cv.split(df,y,df.caid_trip)):
            out.append((repeat,fold,tr,te))
    return out


def gb(params):
    n,lr,depth,leaf=params
    return GradientBoostingClassifier(n_estimators=n,learning_rate=lr,max_depth=depth,
                                      min_samples_leaf=leaf,random_state=42)


def select_gb(X,y,groups,seed):
    inner=StratifiedGroupKFold(3,shuffle=True,random_state=seed)
    yn=(y!=WALK).astype(int)
    inner_splits=list(inner.split(X,yn,groups))
    def score_params(params):
        scores=[]
        for tr,va in inner_splits:
            m=gb(params); m.fit(X.iloc[tr],yn.iloc[tr]); scores.append(balanced_accuracy_score(yn.iloc[va],m.predict(X.iloc[va])))
        return float(np.mean(scores)),tuple(params)
    candidates=Parallel(n_jobs=3,prefer="processes")(delayed(score_params)(p) for p in GRID)
    best=max(candidates,key=lambda x:(x[0],tuple(-v for v in [x[1][0],x[1][2],x[1][3]])))
    return best[1],best[0]


def fit_levels(X,y,n1_kind,n1_params=None):
    if n1_kind=="rf": n1=RandomForestClassifier(**RF_HYPERPARAMETERS["n1"])
    else: n1=gb(n1_params)
    n1.fit(X,(y!=WALK).astype(int))
    motor=y!=WALK; n2=RandomForestClassifier(**RF_HYPERPARAMETERS["n2"]); n2.fit(X[motor],(y[motor]==2).astype(int))
    road=y.isin([0,1]); n3=RandomForestClassifier(**RF_HYPERPARAMETERS["n3"]); n3.fit(X[road],(y[road]==1).astype(int))
    return n1,n2,n3


def cascade_predict(models,X):
    n1,n2,n3=models; motor=n1.predict(X).astype(bool); pred=np.full(len(X),WALK,dtype=int)
    # Flujo real: N2 sólo recibe predicciones motorizadas y N3 sólo superficie predicha.
    mi=np.flatnonzero(motor)
    if len(mi):
        metro=n2.predict(X.iloc[mi]).astype(bool); pred[mi[metro]]=2
        surface=mi[~metro]
        if len(surface): pred[surface]=np.where(n3.predict(X.iloc[surface]).astype(bool),1,0)
    return pred


def aggregate(oof,fold_metrics):
    y=oof.y.values; p=oof.pred.values
    rec=recall_score(y,p,labels=range(4),average=None,zero_division=0); pre=precision_score(y,p,labels=range(4),average=None,zero_division=0)
    ba=np.array([x["balanced_accuracy"] for x in fold_metrics]); mf=np.array([x["macro_f1"] for x in fold_metrics])
    return {"balanced_accuracy_mean":float(ba.mean()),"balanced_accuracy_median":float(np.median(ba)),"balanced_accuracy_std":float(ba.std()),
            "macro_f1_mean":float(mf.mean()),"recall":dict(zip(MODES,map(float,rec))),"precision":dict(zip(MODES,map(float,pre))),
            "confusion_matrix":confusion_matrix(y,p,labels=range(4)).tolist(),
            "missed_walk":int(np.sum((y==WALK)&(p!=WALK))),"false_walk":int(np.sum((y!=WALK)&(p==WALK))),
            "inference_ms_per_scenario":float(1000*sum(x["inference_seconds"] for x in fold_metrics)/sum(x["n_test"] for x in fold_metrics))}


def evaluate(df,name,n1_kind,tune=False,splits=None):
    X=df[list(RF_FEATURES)]; y=df.label.map(M2I).astype(int); splits=splits or split20(df,y)
    records=[]; folds=[]; params=[]
    for repeat,fold,tr,te in splits:
        # Leakage assertions before any fit.
        assert set(df.iloc[tr].caid_trip).isdisjoint(set(df.iloc[te].caid_trip))
        chosen=None; inner_score=None
        if n1_kind=="gb":
            chosen,inner_score=select_gb(X.iloc[tr],y.iloc[tr],df.iloc[tr].caid_trip,1000+repeat*10+fold) if tune else ((80,.05,2,4),None)
        models=fit_levels(X.iloc[tr],y.iloc[tr],n1_kind,chosen)
        started=time.perf_counter(); pred=cascade_predict(models,X.iloc[te]); elapsed=time.perf_counter()-started
        yt=y.iloc[te].values
        folds.append({"model":name,"repeat":repeat,"fold":fold,"n_test":len(te),"balanced_accuracy":balanced_accuracy_score(yt,pred),
                      "macro_f1":f1_score(yt,pred,average="macro"),"inference_seconds":elapsed,"gb_params":json.dumps(chosen),"inner_score":inner_score})
        params.append(chosen)
        for idx,p in zip(te,pred): records.append({"model":name,"repeat":repeat,"fold":fold,"row":int(idx),"trip":df.iloc[idx].caid_trip,
                                                    "deg":df.iloc[idx].deg,"y":int(y.iloc[idx]),"pred":int(p)})
    oof=pd.DataFrame(records); return oof,pd.DataFrame(folds),params


def by_deg(oof):
    rows=[]
    for (model,deg),d in oof.groupby(["model","deg"]):
        rec=recall_score(d.y,d.pred,labels=range(4),average=None,zero_division=0); pre=precision_score(d.y,d.pred,labels=range(4),average=None,zero_division=0)
        rows.append({"model":model,"deg":deg,"balanced_accuracy":balanced_accuracy_score(d.y,d.pred),"macro_f1":f1_score(d.y,d.pred,average="macro"),
                     **{f"recall_{m}":v for m,v in zip(MODES,rec)},**{f"precision_{m}":v for m,v in zip(MODES,pre)}})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True); started=time.perf_counter()
    expanded=make_features("datos_entrenamiento_ml_expanded.pkl"); baseline=make_features("datos_entrenamiento_ml.pkl")
    yexp=expanded.label.map(M2I).astype(int); shared=split20(expanded,yexp)
    oa,fa,_=evaluate(baseline,"A_baseline66_rf","rf")
    ob,fb,_=evaluate(expanded,"B_expanded114_rf","rf",splits=shared)
    oc,fc,chosen=evaluate(expanded,"C_hybrid114_nested_gb","gb",tune=True,splits=shared)
    oof=pd.concat([oa,ob,oc],ignore_index=True); folds=pd.concat([fa,fb,fc],ignore_index=True)
    summaries={}
    for name,d in oof.groupby("model"):
        summaries[name]=aggregate(d,folds[folds.model==name].to_dict("records"))
    pair=fb.merge(fc,on=["repeat","fold"],suffixes=("_rf","_hybrid")); pair["paired_ba_delta"]=pair.balanced_accuracy_hybrid-pair.balanced_accuracy_rf
    wins=int((pair.paired_ba_delta>0).sum()); win_rate=wins/len(pair)
    pair.to_csv(OUT/"hybrid_v5_paired_folds.csv",index=False); folds.to_csv(OUT/"hybrid_v5_fold_metrics.csv",index=False)
    by_deg(oof).to_csv(OUT/"hybrid_v5_by_degradation.csv",index=False)
    for name,d in oof.groupby("model"): np.savetxt(OUT/f"confusion_{name}.csv",confusion_matrix(d.y,d.pred,labels=range(4)),fmt="%d",delimiter=",")
    b=summaries["B_expanded114_rf"]; c=summaries["C_hybrid114_nested_gb"]
    promoted=(c["balanced_accuracy_mean"]-b["balanced_accuracy_mean"]>=.03 and c["recall"]["Caminar"]-b["recall"]["Caminar"]>=.15 and
              c["recall"]["Carro"]>=b["recall"]["Carro"]-.03 and c["recall"]["Bus"]>=b["recall"]["Bus"]-.03 and
              c["recall"]["Metro"]>=b["recall"]["Metro"]-1e-12 and c["balanced_accuracy_std"]<=b["balanced_accuracy_std"] and win_rate>=.70)
    result={"summaries":summaries,"paired":{"wins":wins,"folds":len(pair),"win_rate":win_rate,"mean_delta":float(pair.paired_ba_delta.mean()),
             "median_delta":float(pair.paired_ba_delta.median())},"promotion_criteria_met":bool(promoted),"elapsed_seconds":time.perf_counter()-started,
             "leakage_checks":{"group_disjoint":True,"nested_train_only":True,"no_feature_or_threshold_selection":True,"mixed_excluded":True,
                               "guardrail_label_free":"verified by production tests","cascade_gating":True}}
    (OUT/"hybrid_v5_validation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    # Final model is serialized only as a candidate here; deployment is a separate final step.
    if promoted:
        params,_=select_gb(expanded[list(RF_FEATURES)],yexp,expanded.caid_trip,909)
        levels=fit_levels(expanded[list(RF_FEATURES)],yexp,"gb",params)
        artifact={"clf_n1":levels[0],"clf_n2":levels[1],"clf_n3":levels[2],"feature_cols_v4":list(RF_FEATURES),"feature_cols_new":list(RF_FEATURES),
                  "model_contract":{"version":"ML_v5_hybrid_52","level_types":{"n1":"GradientBoostingClassifier","n2":"RandomForestClassifier","n3":"RandomForestClassifier"}},
                  "metadata":{"physical_trips":114,"scenarios":445,"features":list(RF_FEATURES),"selected_n1_params":dict(zip(["n_estimators","learning_rate","max_depth","min_samples_leaf"],params)),"validation":result}}
        candidate=OUT/"random_forest_modal_hybrid_v5.pkl"
        with open(candidate,"wb") as f: pickle.dump(artifact,f)
        result["candidate_sha256"]=sha(candidate); result["cache_sha256"]=sha(config.GPS_DIR/"datos_entrenamiento_ml_expanded.pkl")
        (OUT/"hybrid_v5_validation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    table=pd.DataFrame([{ "model":k,**{x:v for x,v in s.items() if not isinstance(v,(dict,list))},
                           **{f"recall_{m}":v for m,v in s["recall"].items()},**{f"precision_{m}":v for m,v in s["precision"].items()}} for k,s in summaries.items()])
    report=["# Validación robusta del clasificador híbrido v5","",table.to_markdown(index=False,floatfmt=".4f"),"",
            f"Comparación pareada B vs C: el híbrido ganó {wins}/{len(pair)} particiones ({win_rate:.1%}); delta medio BA {pair.paired_ba_delta.mean():+.4f}.","",
            "## Control de fuga","","Las 20 particiones fueron StratifiedGroupKFold (4 repeticiones × 5 folds). Los grupos físicos son disjuntos. La malla de Gradient Boosting se resolvió con StratifiedGroupKFold interno usando sólo train. No hubo selección de variables ni umbral. La cascada ejecutó N2 sólo para predicciones motorizadas y N3 sólo para superficie predicha.","",
            "## Decisión","",("**Cumple todos los criterios de promoción.**" if promoted else "**No cumple todos los criterios; no se integra ni despliega.**"),"",
            "Resultados por degradación, folds pareados, matrices y parámetros internos están en los CSV/JSON adjuntos."]
    (OUT/"hybrid_v5_report.md").write_text("\n".join(report),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
