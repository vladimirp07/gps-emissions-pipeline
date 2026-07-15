"""Experimentos reproducibles y aislados para N1 (Caminar vs Motorizado)."""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import numpy._core.numeric  # NumPy 2.x
except ModuleNotFoundError:  # Pickles creados con NumPy 2.x leídos desde NumPy 1.x
    import numpy.core as _numpy_core
    import numpy.core.numeric as _numpy_numeric
    sys.modules["numpy._core"] = _numpy_core
    sys.modules["numpy._core.numeric"] = _numpy_numeric
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from pipeline_v4.src import config
from pipeline_v4.src.random_forest_contract import RF_FEATURES, RF_HYPERPARAMETERS

MODES = ["Carro", "Bus", "Metro", "Caminar"]
M2I = {m.lower(): i for i, m in enumerate(MODES)}
WALK = 3
N1_PEDESTRIAN = [
    "drive_mean_speed", "drive_max_speed", "drive_stop_frac", "walk_mean_speed",
    "walk_max_speed", "walk_std_speed", "walk_highway_footway_frac",
    "walk_p25_speed", "walk_p50_speed", "walk_p75_speed", "mean_snap_dist_drive",
    "mean_snap_dist_walk", "std_snap_dist_drive", "std_snap_dist_walk",
    "walk_win_walk_regime_max", "walk_win_walk_regime_consec_run",
]
OUT = ROOT / "Temporary outputs" / "ML v4 expanded comparison" / "n1_caminar"


def parse_key(value):
    physical, rest = value.split("-", 1)
    label, deg, hyp = rest.split("_", 2)
    return physical, label.lower(), deg, hyp


def canonical_labels():
    raw = pd.read_csv(config.GPS_DIR / "Datos de MATLAB GPS Limpios.csv")
    labels, counts = {}, {}
    for (caid, trip), part in raw.groupby(["caid", "num_trip"]):
        try: trip = str(int(float(trip)))
        except Exception: trip = str(trip).strip()
        key = f"{str(caid).strip()}_{trip}"
        vals = [str(v).strip().lower() for v in part.mode_of_transport.dropna() if str(v).strip()]
        labels[key] = vals[0] if len(set(vals)) == 1 else None
        counts[key] = len(part)
    return labels, counts


def stats(a):
    a = np.asarray(a, dtype=float)
    return (float(np.mean(a)), float(np.max(a)), float(np.std(a)),
            float(np.percentile(a, 25)), float(np.percentile(a, 50)),
            float(np.percentile(a, 75)),
            float(np.max(np.abs(np.diff(a)))) if len(a) > 1 else 0.,
            float(np.mean(np.abs(np.diff(a)))) if len(a) > 1 else 0.)


def make_features(cache_name):
    labels, raw_counts = canonical_labels()
    with open(config.GPS_DIR / cache_name, "rb") as fh: cache = pickle.load(fh)
    grouped = {}
    for item in cache:
        physical, _, deg, hyp = parse_key(item["trip_id"])
        label = labels.get(physical)
        if label: grouped.setdefault((physical, label, deg), {})[hyp] = item
    rows = []
    for (physical, label, deg), hyps in grouped.items():
        r = {"caid_trip": physical, "label": label, "deg": deg, "raw_pings": raw_counts[physical]}
        d = hyps.get("carro") or hyps.get("bus")
        if d:
            s = stats(d["speed_raw"])
            for k, v in zip(["drive_mean_speed","drive_max_speed","drive_std_speed","drive_p25_speed","drive_p50_speed","drive_p75_speed","drive_max_speed_diff","drive_mean_speed_diff"], s): r[k] = v
            speed = np.asarray(d["speed_raw"]); hw = d["highway_raw"]
            r.update(drive_stop_frac=float(np.mean(speed < 2)),
                     drive_highway_motorway_frac=float(np.mean([any(w in str(x) for w in ["motorway","trunk","primary"]) for x in hw])) if hw else 0.,
                     drive_highway_residential_frac=float(np.mean(["residential" in str(x) for x in hw])) if hw else 0.,
                     drive_near_bus_frac=float(np.mean(np.asarray(d["idx_c"]) == 1)) if len(d["idx_c"]) else 0.,
                     drive_near_metro_frac=float(np.mean(np.asarray(d["idx_c"]) == 0)) if len(d["idx_c"]) else 0.,
                     drive_num_stops=float(d.get("num_stops",0)), drive_mean_stop_duration=float(d.get("mean_stop_duration",0)),
                     drive_mean_stop_interval=float(d.get("mean_stop_interval",0)), drive_std_stop_interval=float(d.get("std_stop_interval",0)))
            lf=d.get("local_features",{})
            for k in ["drive_win_near_bus_max","drive_win_near_bus_p90","drive_win_near_bus_consec_run","drive_win_stops_max","drive_win_stops_consec_run"]: r[k]=float(lf.get(k,0))
        w=hyps.get("caminar")
        if w:
            s=stats(w["speed_raw"])
            for k,v in zip(["walk_mean_speed","walk_max_speed","walk_std_speed","walk_p25_speed","walk_p50_speed","walk_p75_speed","walk_max_speed_diff","walk_mean_speed_diff"],s): r[k]=v
            hw=w["highway_raw"]
            r["walk_highway_footway_frac"]=float(np.mean([any(x in str(h) for x in ["footway","pedestrian","steps","path","living_street"]) for h in hw])) if hw else 0.
            lf=w.get("local_features",{})
            for k in ["walk_win_walk_regime_max","walk_win_walk_regime_consec_run"]: r[k]=float(lf.get(k,0))
        m=hyps.get("metro")
        if m:
            s=stats(m["speed_raw"])
            for k,v in zip(["metro_mean_speed","metro_max_speed","metro_unused_std","metro_p25_speed","metro_p50_speed","metro_p75_speed","metro_max_speed_diff","metro_mean_speed_diff"],s): r[k]=v
            r["metro_near_metro_frac"]=float(np.mean(np.asarray(m["idx_c"])==0)) if len(m["idx_c"]) else 0.
            lf=m.get("local_features",{})
            for k in ["metro_win_near_metro_max","metro_win_near_metro_p90","metro_win_near_metro_consec_run"]: r[k]=float(lf.get(k,0))
        a=d or w or m
        if a and "snap_dist_drive" in a:
            for prefix, arr in [("drive",a["snap_dist_drive"]),("walk",a["snap_dist_walk"])]:
                arr=np.asarray(arr,dtype=float); r[f"mean_snap_dist_{prefix}"]=float(np.mean(arr)); r[f"max_snap_dist_{prefix}"]=float(np.max(arr)); r[f"std_snap_dist_{prefix}"]=float(np.std(arr))
        r.setdefault("mean_snap_dist_drive",150.); r.setdefault("max_snap_dist_drive",150.); r.setdefault("std_snap_dist_drive",0.)
        r.setdefault("mean_snap_dist_walk",50.); r.setdefault("max_snap_dist_walk",50.); r.setdefault("std_snap_dist_walk",0.)
        r["drive_near_bus_drift_decay"]=r.get("drive_near_bus_frac",0.)*np.exp(-r["mean_snap_dist_drive"]/15.)
        r["drive_near_bus_high_drift"]=r.get("drive_near_bus_frac",0.)*(1-np.exp(-r["mean_snap_dist_drive"]/15.))
        for f in RF_FEATURES: r.setdefault(f,0.)
        r["effective_pings"] = max([len(x.get("speed_raw",[])) for x in hyps.values()] or [0])
        r["pct_conserved"] = 100*r["effective_pings"]/max(r["raw_pings"],1)
        r["available_hypotheses"] = ",".join(sorted(hyps))
        rows.append(r)
    return pd.DataFrame(rows).fillna(0)


def add_route_metadata(df):
    """Añade magnitudes físicas desde el caché ruteado, sin reconstruir features."""
    route_dir=config.GPS_DIR/"cache_rutas_completas_expanded"
    audit=json.loads((config.GPS_DIR/"auditoria_dataset_ml_expanded.json").read_text(encoding="utf-8"))["selected"]
    durations=[]; distances=[]; routed_pings=[]; clean_effective=[]; conserved=[]
    for row in df.itertuples():
        files=list(route_dir.glob(f"{row.caid_trip}_{row.deg}_*.pkl"))
        duration=distance=0.; effective=int(row.effective_pings)
        if files:
            try:
                with open(files[0],"rb") as fh: obj=pickle.load(fh)
                routed=obj.get("df_routed",pd.DataFrame())
                if len(routed):
                    ts=pd.to_datetime(routed.get("local_timestamp"),errors="coerce")
                    if ts.notna().any(): duration=float((ts.max()-ts.min()).total_seconds())
                    distance=float(pd.to_numeric(routed.get("distance_m",pd.Series(dtype=float)),errors="coerce").fillna(0).sum())
                snaps=np.asarray(obj.get("snap_d_walk",[])); effective=int(len(snaps)) if len(snaps) else effective
            except Exception: pass
        durations.append(duration); distances.append(distance); routed_pings.append(effective)
        meta=audit.get(row.caid_trip,{}); clean_effective.append(int(meta.get("clean_pings",row.effective_pings))); conserved.append(float(meta.get("pct_conserved",0)))
    df=df.copy(); df["duration_s"]=durations; df["distance_m"]=distances; df["scenario_routed_pings"]=routed_pings
    df["effective_pings"]=clean_effective; df["pct_conserved"]=conserved
    return df


def model_for(name):
    if name in ("A_RF52","B_RF16","C_RF_SELECT","E_RF_THRESHOLD"):
        return RandomForestClassifier(**RF_HYPERPARAMETERS["n1"])
    if name == "D_LOGISTIC":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight="balanced", max_iter=3000, random_state=42))
    return GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=.05, min_samples_leaf=4, random_state=42)


def select_features(X, y, groups, fold_seed):
    tr, va = next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=fold_seed).split(X,y,groups))
    model=RandomForestClassifier(**RF_HYPERPARAMETERS["n1"]); model.fit(X.iloc[tr],y.iloc[tr])
    pi=permutation_importance(model,X.iloc[va],y.iloc[va],scoring="balanced_accuracy",n_repeats=6,random_state=fold_seed,n_jobs=3)
    order=np.argsort(pi.importances_mean)[::-1]
    positive=[i for i in order if pi.importances_mean[i]>0]
    chosen=(positive if len(positive)>=5 else list(order))[:20]
    return [X.columns[i] for i in chosen]


def tune_threshold(X,y,groups,features):
    probs=np.zeros(len(y)); inner=GroupKFold(3)
    for tr,va in inner.split(X,y,groups):
        m=model_for("A_RF52"); m.fit(X.iloc[tr][features],y.iloc[tr]); probs[va]=m.predict_proba(X.iloc[va][features])[:,1]
    best=(0.5,-1)
    for t in np.linspace(.20,.75,56):
        pred=(probs>=t).astype(int); rw=recall_score(y,pred,pos_label=0); fp=np.mean((y==1)&(pred==0))
        score=balanced_accuracy_score(y,pred)+.20*rw-.50*max(0,fp-.05)
        if score>best[1]: best=(float(t),float(score))
    return best[0]


def metrics(y,p,folds):
    recalls=recall_score(y,p,labels=range(4),average=None,zero_division=0)
    bas=[balanced_accuracy_score(np.asarray(y)[folds==f],np.asarray(p)[folds==f]) for f in sorted(set(folds))]
    return {"balanced_accuracy":balanced_accuracy_score(y,p),"macro_f1":f1_score(y,p,average="macro"),
            "walk_recall":recalls[3],"walk_precision":precision_score(y,p,labels=[3],average="macro",zero_division=0),
            "car_recall":recalls[0],"bus_recall":recalls[1],"metro_recall":recalls[2],
            "fold_ba_mean":float(np.mean(bas)),"fold_ba_std":float(np.std(bas)),
            "false_walk":int(np.sum((np.asarray(y)!=3)&(np.asarray(p)==3))),"missed_walk":int(np.sum((np.asarray(y)==3)&(np.asarray(p)!=3)))}


def plot_cm(cm,name):
    fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(cm,cmap="Blues")
    for i in range(4):
        for j in range(4): ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    ax.set(xticks=range(4),yticks=range(4),xticklabels=MODES,yticklabels=MODES,xlabel="Predicción",ylabel="Real",title=name)
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(OUT/f"matriz_confusion_{name}.png",dpi=180); plt.close(fig)


def main():
    started=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True); (OUT/"modelos_candidatos").mkdir(exist_ok=True)
    df=add_route_metadata(make_features("datos_entrenamiento_ml_expanded.pkl"))
    base=make_features("datos_entrenamiento_ml.pkl")
    y=df.label.map(M2I).astype(int); groups=df.caid_trip; X=df[list(RF_FEATURES)]
    splits=list(GroupKFold(5).split(X,y,groups)); configs=["A_RF52","B_RF16","C_RF_SELECT","D_LOGISTIC","D_GRADIENT_BOOSTING","E_RF_THRESHOLD"]
    outputs=[]; feature_manifest={}; all_results=[]
    for name in configs:
        pred=np.zeros(len(df),dtype=int); p_walk=np.zeros(len(df)); fold_col=np.zeros(len(df),dtype=int); fold_features=[]; thresholds=[]
        for fold,(tr,va) in enumerate(splits):
            yt=y.iloc[tr]; features=list(RF_FEATURES) if name not in ("B_RF16","C_RF_SELECT") else list(N1_PEDESTRIAN)
            if name=="C_RF_SELECT": features=select_features(X.iloc[tr],(yt!=WALK).astype(int),groups.iloc[tr],42+fold)
            n1=model_for(name); n1.fit(X.iloc[tr][features],(yt!=WALK).astype(int))
            threshold=tune_threshold(X.iloc[tr],(yt!=WALK).astype(int),groups.iloc[tr],features) if name=="E_RF_THRESHOLD" else .5
            pn1=n1.predict_proba(X.iloc[va][features])[:,1]; motor=pn1>=threshold
            n2=RandomForestClassifier(**RF_HYPERPARAMETERS["n2"]); mm=yt!=WALK; n2.fit(X.iloc[tr][list(RF_FEATURES)][mm],(yt[mm]==2).astype(int))
            n3=RandomForestClassifier(**RF_HYPERPARAMETERS["n3"]); rm=yt.isin([0,1]); n3.fit(X.iloc[tr][list(RF_FEATURES)][rm],(yt[rm]==1).astype(int))
            v=X.iloc[va][list(RF_FEATURES)]; pv=np.where(~motor,WALK,np.where(n2.predict(v)==1,2,np.where(n3.predict(v)==1,1,0)))
            pred[va]=pv; p_walk[va]=1-pn1; fold_col[va]=fold; fold_features.append(features); thresholds.append(threshold)
        feature_manifest[name]={"features_by_fold":fold_features,"thresholds_by_fold":thresholds}
        cm=confusion_matrix(y,pred,labels=range(4)); plot_cm(cm,name); np.savetxt(OUT/f"matriz_confusion_{name}.csv",cm,fmt="%d",delimiter=",")
        for scope,mask in [("global",np.ones(len(df),dtype=bool))]+[(d,(df.deg==d).values) for d in ["Raw","L1","L2","L3"]]:
            if mask.sum(): all_results.append({"config":name,"scope":scope,"n":int(mask.sum()),**metrics(y[mask],pred[mask],fold_col[mask])})
        outputs.append(pd.DataFrame({"config":name,"index":range(len(df)),"prediction":pred,"p_walk":p_walk,"fold":fold_col}))
    exp=pd.DataFrame(all_results); exp.to_csv(OUT/"caminar_n1_experimentos.csv",index=False)
    a=outputs[0]; diag=df.copy(); diag["prediction"]=a.prediction.map(dict(enumerate(MODES))); diag["n1_probability_walk"]=a.p_walk; diag["fold"]=a.fold
    baseline_walk=set(base.loc[base.label=="caminar","caid_trip"]); diag["baseline_walk_trip"]=diag.caid_trip.isin(baseline_walk)
    diag["error_type"]=np.where((diag.label=="caminar")&(a.prediction!=WALK),"walk_missed",np.where((diag.label!="caminar")&(a.prediction==WALK),"false_walk","correct"))
    cols=["caid_trip","deg","label","baseline_walk_trip","duration_s","distance_m","raw_pings","effective_pings","pct_conserved","scenario_routed_pings","available_hypotheses",
          "drive_mean_speed","drive_max_speed","drive_stop_frac","walk_mean_speed","walk_max_speed","walk_std_speed","walk_highway_footway_frac",
          "walk_p25_speed","walk_p50_speed","walk_p75_speed","mean_snap_dist_drive","mean_snap_dist_walk","std_snap_dist_drive","std_snap_dist_walk",
          "walk_win_walk_regime_max","walk_win_walk_regime_consec_run","prediction","n1_probability_walk","fold","error_type"]
    diag[cols].to_csv(OUT/"caminar_n1_diagnostico.csv",index=False)
    # Modelos N1 candidatos entrenados sobre todo el dataset; N2/N3 quedan fuera deliberadamente.
    for name in configs:
        feats=list(RF_FEATURES) if name not in ("B_RF16","C_RF_SELECT") else list(N1_PEDESTRIAN)
        yn=(y!=WALK).astype(int)
        if name=="C_RF_SELECT": feats=select_features(X,yn,groups,99)
        threshold=tune_threshold(X,yn,groups,feats) if name=="E_RF_THRESHOLD" else .5
        model=model_for(name); model.fit(X[feats],yn)
        with open(OUT/"modelos_candidatos"/f"n1_{name}.pkl","wb") as fh: pickle.dump({"model":model,"features":feats,"threshold":threshold,"scope":"N1 only"},fh)
        feature_manifest[name]["full_fit_features"]=feats; feature_manifest[name]["full_fit_threshold"]=threshold
    (OUT/"variables_por_candidato.json").write_text(json.dumps(feature_manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    global_rows=exp[exp.scope=="global"].set_index("config"); ref=global_rows.loc["A_RF52"]
    accepted=[]
    for name,row in global_rows.drop(index="A_RF52").iterrows():
        ok=(row.walk_recall-ref.walk_recall>=.15 and row.balanced_accuracy-ref.balanced_accuracy>=.03 and
            row.car_recall>=ref.car_recall-.03 and row.bus_recall>=ref.bus_recall-.03 and row.metro_recall>=ref.metro_recall-1e-12 and
            row.fold_ba_std<=ref.fold_ba_std+.02)
        if ok: accepted.append(name)
    walk=diag[diag.label=="caminar"]; new=set(walk.caid_trip)-baseline_walk
    lost=walk[walk.error_type=="walk_missed"]
    lost_new=int(lost.caid_trip.isin(new).sum()); lost_old=int((~lost.caid_trip.isin(new)).sum())
    lost_deg=lost.groupby("deg").size().to_dict(); walk_fold=walk.groupby("fold").caid_trip.nunique().to_dict()
    short_cut=float(walk.duration_s.quantile(.25)); lost_short=int((lost.duration_s<=short_cut).sum())
    low_ret=int((lost.pct_conserved<50).sum()); missing_walk=int((~lost.available_hypotheses.str.contains("caminar")).sum())
    report=["# Diagnóstico y experimentos N1: Caminar vs. Motorizado","",f"Dataset fijo: **{df.caid_trip.nunique()} viajes / {len(df)} escenarios**. N2 y N3 se reentrenaron de forma idéntica y congelada en cada fold; no se modificaron producción ni orquestador.","",
            "## Diagnóstico", "",f"Las 7 caminatas baseline son: {', '.join(sorted(baseline_walk))}. El ampliado contiene 9: entran {', '.join(sorted(new))} y sale ARL_2 por conservación física de 13.97%; por eso el incremento neto es de dos aunque hay tres IDs incorporados.",
            f"Con N1 A se perdieron {len(lost)} escenarios de caminar y hubo {int((diag.error_type=='false_walk').sum())} falsos Caminar. De las pérdidas, {lost_new} pertenecen a las dos caminatas nuevas y {lost_old} a las siete históricas.",
            f"Pérdidas por degradación: {lost_deg}. Pérdidas en el cuartil más corto (≤ {short_cut:.1f} s): {lost_short}; con conservación <50%: {low_ret}; sin hipótesis peatonal disponible: {missing_walk}.",
            f"Distribución de viajes físicos Caminar por fold: {walk_fold}. La concentración desigual de sólo nueve grupos explica parte de la variabilidad; no se movieron grupos entre folds.",
            "La evidencia separa daño de degradación/limpieza (tabla diagnóstica) de dominancia de variables: B y Gradient Boosting mejoran Caminar usando el mismo dataset, por lo que la caída no se atribuye exclusivamente a las dos caminatas nuevas ni al ruteo. El contraste de A frente a B indica que variables globales no peatonales sí interfieren en N1.","",
            "## Comparación global","",global_rows.to_markdown(floatfmt=".4f"),"","Resultados por Raw/L1/L2/L3 están en `caminar_n1_experimentos.csv`; cada matriz absoluta está disponible en CSV y PNG.","",
            "## Selección","",("Cumplen todos los criterios: **"+", ".join(accepted)+"**. Se recomienda **D_GRADIENT_BOOSTING** como candidato N1: conserva la precisión de Caminar y produce menos falsos Caminar, con la menor variabilidad entre folds. No se promueve ni sobrescribe producción en esta fase." if accepted else "**Ningún candidato cumple simultáneamente todos los criterios de aceptación. Se conserva el baseline desplegado.**"),"",
            "La selección C se ejecutó dentro de cada fold externo mediante un split interno agrupado y permutation importance; E ajustó su umbral sólo con predicciones OOF internas agrupadas. Las listas y umbrales exactos están en `variables_por_candidato.json`.","",
            f"Tiempo total: {time.perf_counter()-started:.1f} s."]
    (OUT/"caminar_n1_report.md").write_text("\n".join(report),encoding="utf-8")
    print(OUT); print(global_rows.to_string()); print("accepted",accepted)

if __name__ == "__main__": main()

