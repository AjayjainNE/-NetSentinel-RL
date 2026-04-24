"""
NetSentinel-RL — Data Preprocessing Pipeline
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import logging, pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FLOW_FEATURES = [
    "Flow Duration","Total Fwd Packets","Total Backward Packets",
    "Total Length of Fwd Packets","Fwd Packet Length Max","Fwd Packet Length Mean",
    "Bwd Packet Length Max","Flow Bytes/s","Flow Packets/s",
    "Flow IAT Mean","Flow IAT Std","Fwd IAT Mean","Bwd IAT Mean",
    "Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags",
    "SYN Flag Count","RST Flag Count","ACK Flag Count",
    "Down/Up Ratio","Avg Packet Size",
]

ATTACK_LABELS = {
    "BENIGN":0,"DoS Hulk":1,"DoS GoldenEye":1,"DoS slowloris":1,"DoS Slowhttptest":1,
    "DDoS":2,"PortScan":3,"FTP-Patator":4,"SSH-Patator":4,"Bot":5,
    "Web Attack":6,"Web Attack – Brute Force":6,"Infiltration":7,
}

ATTACK_NAMES = {0:"Benign",1:"DoS",2:"DDoS",3:"PortScan",4:"BruteForce",5:"Botnet",6:"WebAttack",7:"Infiltration"}


class NetworkFlowTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, clip_percentile=99.5):
        self.clip_percentile = clip_percentile
        self.clip_values_ = {}
        self.feature_names_ = []

    def fit(self, X, y=None):
        df = self._sanitise(X.copy())
        for col in df.columns:
            self.clip_values_[col] = np.percentile(df[col], self.clip_percentile)
        self.feature_names_ = list(df.columns) + self._derived_names()
        return self

    def transform(self, X, y=None):
        df = self._sanitise(X.copy())
        for col, v in self.clip_values_.items():
            if col in df.columns: df[col] = df[col].clip(upper=v)
        return self._add_derived(df).values.astype(np.float32)

    def _sanitise(self, df):
        return df.replace([np.inf,-np.inf], np.nan).fillna(0)

    def _add_derived(self, df):
        eps=1e-9
        df["packet_ratio"] = df["Total Fwd Packets"]/(df["Total Backward Packets"]+eps)
        df["byte_ratio"]   = df["Total Length of Fwd Packets"]/(df["Fwd Packet Length Max"]+eps)
        df["iat_cv"]       = df["Flow IAT Std"]/(df["Flow IAT Mean"]+eps)
        df["flag_density"] = (df["SYN Flag Count"]+df["RST Flag Count"]+df["ACK Flag Count"])/(df["Total Fwd Packets"]+df["Total Backward Packets"]+eps)
        df["duration_log"] = np.log1p(df["Flow Duration"])
        df["bytes_per_pkt"]= df["Flow Bytes/s"]/(df["Flow Packets/s"]+eps)
        return df

    def _derived_names(self):
        return ["packet_ratio","byte_ratio","iat_cv","flag_density","duration_log","bytes_per_pkt"]

    def get_feature_names_out(self):
        return self.feature_names_


class FlowTextEncoder:
    def encode_flow(self, row):
        return (
            f"flow: duration={int(row.get('Flow Duration',0))}ms "
            f"packets_fwd={int(row.get('Total Fwd Packets',0))} "
            f"packets_bwd={int(row.get('Total Backward Packets',0))} "
            f"bytes_rate={int(row.get('Flow Bytes/s',0))} "
            f"flags: syn={int(row.get('SYN Flag Count',0))} "
            f"rst={int(row.get('RST Flag Count',0))} "
            f"ack={int(row.get('ACK Flag Count',0))} "
            f"psh={int(row.get('Fwd PSH Flags',0))} "
            f"urg={int(row.get('Fwd URG Flags',0))} "
            f"stats: iat_mean={int(row.get('Flow IAT Mean',0))}ms "
            f"pkt_size={int(row.get('Avg Packet Size',0))}B "
            f"ratio={round(row.get('Down/Up Ratio',1.0),2)}"
        )
    def encode_batch(self, df):
        return [self.encode_flow(row) for _,row in df.iterrows()]


def build_preprocessing_pipeline():
    return Pipeline([("flow_transform",NetworkFlowTransformer(99.5)),("scaler",StandardScaler())])


def load_and_preprocess(data_path="data/raw/cicids2017/synthetic_cicids.parquet", test_size=0.2, random_state=42):
    path=Path(data_path)
    df = pd.read_parquet(path) if path.suffix==".parquet" else pd.read_csv(path)
    df.columns = df.columns.str.strip()
    available = [f for f in FLOW_FEATURES if f in df.columns]
    X_raw = df[available].copy()
    y_raw = df["Label"].map(ATTACK_LABELS).fillna(0).astype(int)
    X_tmp,X_test,y_tmp,y_test = train_test_split(X_raw,y_raw,test_size=test_size,random_state=random_state,stratify=y_raw)
    X_train,X_val,y_train,y_val = train_test_split(X_tmp,y_tmp,test_size=0.125,random_state=random_state,stratify=y_tmp)
    pipeline = build_preprocessing_pipeline()
    X_train_p=pipeline.fit_transform(X_train)
    X_val_p=pipeline.transform(X_val)
    X_test_p=pipeline.transform(X_test)
    enc=FlowTextEncoder()
    texts_train=enc.encode_batch(X_train.reset_index(drop=True))
    texts_val=enc.encode_batch(X_val.reset_index(drop=True))
    texts_test=enc.encode_batch(X_test.reset_index(drop=True))
    out=Path("data/processed"); out.mkdir(parents=True,exist_ok=True)
    np.save(out/"X_train.npy",X_train_p); np.save(out/"X_val.npy",X_val_p); np.save(out/"X_test.npy",X_test_p)
    np.save(out/"y_train.npy",y_train.values); np.save(out/"y_val.npy",y_val.values); np.save(out/"y_test.npy",y_test.values)
    with open(out/"pipeline.pkl","wb") as f: pickle.dump(pipeline,f)
    feature_names=pipeline.named_steps["flow_transform"].get_feature_names_out()
    log.info(f"Train:{X_train_p.shape} Val:{X_val_p.shape} Test:{X_test_p.shape}")
    return dict(X_train=X_train_p,X_val=X_val_p,X_test=X_test_p,
                y_train=y_train.values,y_val=y_val.values,y_test=y_test.values,
                pipeline=pipeline,feature_names=feature_names,
                texts_train=texts_train,texts_val=texts_val,texts_test=texts_test,
                attack_names=ATTACK_NAMES,n_classes=len(ATTACK_NAMES))

if __name__=="__main__":
    r=load_and_preprocess()
    print(f"Train:{r['X_train'].shape} | Features:{len(r['feature_names'])}")
