"""
app.py — DataQuality Agent v3
Fichier unique : Auth + Engine + Interface Streamlit
Déployer sur Streamlit Cloud avec requirements.txt
"""

# ══════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════
"""
DataQuality Agent — app_v3.py
Streamlit Cloud · Login/Password · 4 étapes · Engine auto (Pandas/PySpark)

Structure :
  app_v3.py      ← Ce fichier (interface)
  engine.py      ← Moteur de scoring (pandas + pyspark)
  auth.py        ← Gestion login/password
  users.json     ← Créé automatiquement au premier lancement
  requirements.txt
"""

import streamlit as st
import secrets as secrets_mod
import string
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import base64, json, re, io
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Imports locaux

# ══════════════════════════════════════════════════════════════
# AUTH — Comptes utilisateurs
# ══════════════════════════════════════════════════════════════
"""
auth.py — Authentification simplifiée
Compte de démonstration hardcodé — pas de base de données.
"""

# ══════════════════════════════════════════════════════════════
# COMPTE DE TEST — modifier ici pour changer les accès
# ══════════════════════════════════════════════════════════════

USERS = {
    "demo": {
        "password": "dataquality2025",
        "role":     "client",
    },
    "admin": {
        "password": "admin1234",
        "role":     "admin",
    },
}


# ══════════════════════════════════════════════════════════════
# API (identique à avant — app_v3.py n'a pas besoin de changer)
# ══════════════════════════════════════════════════════════════

def verify_login(username: str, password: str):
    user = USERS.get(username)
    if user and user["password"] == password:
        return {"username": username, "role": user["role"]}
    return None

def is_logged_in(session_state) -> bool:
    return session_state.get("user") is not None

def get_current_user(session_state):
    return session_state.get("user")

def login(session_state, username: str, password: str) -> bool:
    user = verify_login(username, password)
    if user:
        session_state["user"] = user
        return True
    return False

def logout(session_state):
    session_state["user"] = None
    for key in ["df","result","rules","source_name","source_type",
                "step","freshness_h","alert_t","detected"]:
        if key in session_state:
            del session_state[key]

# ══════════════════════════════════════════════════════════════
# ENGINE — Scoring (Pandas + PySpark)
# ══════════════════════════════════════════════════════════════
"""
engine.py — Moteur de scoring unifié
Auto-détecte PySpark si disponible, sinon bascule sur Pandas.
Importé par app_v3.py ET par le notebook Databricks client.
"""

import pandas as pd
import numpy as np
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

# ── Détection PySpark ─────────────────────────────────────────
try:
    from pyspark.sql import DataFrame as SparkDF
    from pyspark.sql import functions as F
    from pyspark.sql import types as T
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


# ══════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════

@dataclass
class ColumnScore:
    name:         str
    completeness: float = 0.0
    uniqueness:   float = 0.0
    overall:      float = 0.0
    issues:       list  = field(default_factory=list)

@dataclass
class TableScore:
    table_name:      str
    row_count:       int
    col_count:       int
    engine:          str   = "pandas"   # "pandas" ou "pyspark"
    completeness:    float = 0.0
    uniqueness:      float = 0.0
    freshness:       float = 0.0
    consistency:     float = 0.0
    distribution:    float = 0.0
    validity:        float = 0.0
    correlation:     float = 0.0
    volumetry:       float = 0.0
    standardization: float = 0.0
    global_score:    float = 0.0
    columns:         list  = field(default_factory=list)
    issues:          list  = field(default_factory=list)
    scored_at:       str   = field(default_factory=lambda: datetime.now().isoformat())
    custom_rules:    list  = field(default_factory=list)


# ══════════════════════════════════════════════════════════════
# AUTO DETECTOR (pandas — fonctionne partout)
# ══════════════════════════════════════════════════════════════

class ColumnAutoDetector:
    EMAIL_KW  = ["email","mail","courriel"]
    PHONE_KW  = ["phone","tel","mobile","gsm","portable"]
    DATE_KW   = ["date","created_at","updated_at","timestamp",
                 "subscription","since","birth","expir","modified_at","datetime"]
    START_KW  = ["created","start","begin","debut","open","first","from"]
    END_KW    = ["end","fin","expir","close","stop","last","to","until"]
    EMAIL_RE  = re.compile(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$')

    def detect(self, df: pd.DataFrame) -> dict:
        res = {"email_columns":[],"phone_columns":[],"date_columns":[],"correlation_rules":[]}
        for col in df.columns:
            cl = col.lower().strip()
            ct = str(df[col].dtype)
            if any(kw in cl for kw in self.EMAIL_KW):
                res["email_columns"].append(col); continue
            if any(kw in cl for kw in self.PHONE_KW):
                res["phone_columns"].append(col); continue
            if any(kw in cl for kw in self.DATE_KW) or "datetime" in ct:
                res["date_columns"].append(col); continue
            if ct == "object" and len(df[col].dropna()) > 0:
                samp = df[col].dropna().astype(str).sample(min(30, len(df[col].dropna())), random_state=42)
                if pd.to_datetime(samp, errors="coerce").notna().mean() > 0.7:
                    res["date_columns"].append(col); continue
                if samp.str.match(self.EMAIL_RE).mean() > 0.7:
                    res["email_columns"].append(col)

        # Corrélations auto HT/TTC
        num = df.select_dtypes(include=["float64","int64"]).columns.tolist()
        ht  = [c for c in num if "ht"  in c.lower() and "ttc" not in c.lower()]
        ttc = [c for c in num if "ttc" in c.lower()]
        for h, t in zip(ht, ttc):
            res["correlation_rules"].append({
                "col_a": h, "col_b": t, "operator": "<",
                "name": f"{h} < {t}", "severity": "high"
            })

        # Corrélations auto dates début/fin
        dates  = res["date_columns"]
        starts = [c for c in dates if any(kw in c.lower() for kw in self.START_KW)]
        ends   = [c for c in dates if any(kw in c.lower() for kw in self.END_KW)]
        for s, e in zip(starts, ends):
            if s != e:
                res["correlation_rules"].append({
                    "col_a": s, "col_b": e, "operator": "<",
                    "name": f"{s} avant {e}", "severity": "high"
                })
        return res


# ══════════════════════════════════════════════════════════════
# ENGINE PANDAS
# ══════════════════════════════════════════════════════════════

class PandasScorer:
    """Engine pandas — fonctionne partout, jusqu'à ~5M lignes."""

    WEIGHTS = {
        "completeness":0.20,"consistency":0.15,"validity":0.15,
        "uniqueness":0.12,"freshness":0.10,"distribution":0.08,
        "correlation":0.08,"volumetry":0.07,"standardization":0.05,
    }

    def __init__(self, table_name="dataset", date_columns=None, email_columns=None,
                 phone_columns=None, custom_rules=None, correlation_rules=None,
                 freshness_threshold_hours=24):
        self.table_name = table_name
        self.date_columns = date_columns or []
        self.email_columns = email_columns or []
        self.phone_columns = phone_columns or []
        self.custom_rules = custom_rules or []
        self.correlation_rules = correlation_rules or []
        self.freshness_threshold_hours = freshness_threshold_hours

    def score(self, df: pd.DataFrame) -> TableScore:
        r = TableScore(
            table_name=self.table_name,
            row_count=len(df),
            col_count=len(df.columns),
            engine="pandas",
            custom_rules=self.custom_rules,
        )
        r.completeness    = self._completeness(df, r)
        r.uniqueness      = self._uniqueness(df, r)
        r.freshness       = self._freshness(df, r)
        r.consistency     = self._consistency(df, r)
        r.distribution    = self._distribution(df, r)
        r.validity        = self._validity(df, r)
        r.correlation     = self._correlation(df, r)
        r.volumetry       = 80.0  # One-shot : neutre
        r.standardization = self._standardization(df, r)
        r.global_score    = round(sum(getattr(r, d) * w for d, w in self.WEIGHTS.items()), 1)
        r.columns         = self._column_scores(df)
        return r

    def _completeness(self, df, r):
        if df.empty: return 0.0
        for col, pct in (df.isnull().mean() * 100).items():
            if pct > 20:
                r.issues.append({"dimension":"completeness",
                    "severity":"high" if pct > 50 else "medium",
                    "column": col, "message": f"{pct:.1f}% de valeurs nulles"})
        return round((1 - df.isnull().sum().sum() / df.size) * 100, 1)

    def _uniqueness(self, df, r):
        if len(df) < 2: return 100.0
        dup = df.duplicated().sum(); pct = dup / len(df) * 100
        if pct > 5:
            r.issues.append({"dimension":"uniqueness",
                "severity":"high" if pct > 20 else "medium",
                "column":"all", "message": f"{dup:,} lignes dupliquées ({pct:.1f}%)"})
        return round(max(0, 100 - pct * 2), 1)

    def _freshness(self, df, r):
        if not self.date_columns: return 75.0
        scores = []; now = pd.Timestamp.now()
        for col in self.date_columns:
            if col not in df.columns: continue
            try:
                dates = pd.to_datetime(df[col], errors="coerce").dropna()
                if dates.empty: continue
                lag = (now - dates.max()).total_seconds() / 3600
                scores.append(max(0, 100 - (lag / self.freshness_threshold_hours) * 100))
                if lag > self.freshness_threshold_hours:
                    r.issues.append({"dimension":"freshness",
                        "severity":"high" if lag > self.freshness_threshold_hours * 3 else "medium",
                        "column": col, "message": f"Dernière donnée il y a {lag:.0f}h"})
            except: pass
        return round(np.mean(scores), 1) if scores else 75.0

    def _consistency(self, df, r):
        v, c = 0, 0
        for col in df.select_dtypes(include=["float64","int64"]).columns:
            if any(kw in col.lower() for kw in ["price","prix","amount","montant","age","qty","quantity","stock"]):
                neg = (df[col] < 0).sum(); v += neg; c += len(df)
                if neg > 0:
                    r.issues.append({"dimension":"consistency","severity":"high",
                        "column":col,"message":f"{neg:,} valeurs negatives"})
        for rule in self.custom_rules:
            col_name = rule.get("column","")
            rule_op  = rule.get("operator","")
            rule_val = rule.get("value","")
            try:
                # Strategie 1 : df.eval() direct
                n = (~df.eval(rule["condition"])).sum()
                c += len(df); v += n
                if n > 0:
                    r.issues.append({"dimension":"consistency",
                        "severity":rule.get("severity","medium"),
                        "column":col_name or "custom",
                        "message":f"Regle {rule['name']}: {n:,} violations"})
            except Exception:
                try:
                    # Strategie 2 : renommer colonnes pour eval() (espaces, tirets...)
                    col_map = {c2: re.sub(r"[^a-zA-Z0-9_]", "_", c2) for c2 in df.columns}
                    safe_df = df.rename(columns=col_map)
                    safe_cond = rule["condition"]
                    for orig, safe in sorted(col_map.items(), key=lambda x: -len(x[0])):
                        safe_cond = safe_cond.replace(f"", safe).replace(orig, safe)
                    n = (~safe_df.eval(safe_cond)).sum()
                    c += len(df); v += n
                    if n > 0:
                        r.issues.append({"dimension":"consistency",
                            "severity":rule.get("severity","medium"),
                            "column":col_name or "custom",
                            "message":f"Regle {rule['name']}: {n:,} violations"})
                except Exception:
                    try:
                        # Strategie 3 : evaluation directe pandas (fallback)
                        if col_name and col_name in df.columns and rule_op:
                            s = pd.to_numeric(df[col_name], errors="coerce")
                            val_f = float(rule_val)
                            ops = {">": s>val_f, ">=": s>=val_f, "<": s<val_f,
                                   "<=": s<=val_f, "==": s==val_f, "!=": s!=val_f}
                            if rule_op in ops:
                                n = (~ops[rule_op]).sum(); c += len(df); v += n
                                if n > 0:
                                    r.issues.append({"dimension":"consistency",
                                        "severity":rule.get("severity","medium"),
                                        "column":col_name,
                                        "message":f"Regle {rule['name']}: {n:,} violations"})
                    except Exception:
                        r.issues.append({"dimension":"consistency","severity":"low",
                            "column":col_name or "?",
                            "message":f"Regle non applicable sur ce dataset: {rule['name']}"})
        return 90.0 if c == 0 else round(max(0, (1 - v / c) * 100), 1)

    def _distribution(self, df, r):
        cols = df.select_dtypes(include=["float64","int64"]).columns
        if not len(cols): return 90.0
        ratios = []
        for col in cols:
            s = df[col].dropna()
            if len(s) < 10: continue
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75); IQR = Q3 - Q1
            if IQR == 0: continue
            out = ((s < Q1 - 3*IQR) | (s > Q3 + 3*IQR)).sum()
            ratio = out / len(s); ratios.append(ratio)
            if ratio > 0.05:
                r.issues.append({"dimension":"distribution","severity":"medium",
                    "column":col,"message":f"{out:,} outliers extrêmes ({ratio*100:.1f}%)"})
        return 90.0 if not ratios else round(max(0, 100 - np.mean(ratios) * 500), 1)

    def _validity(self, df, r):
        v, c = 0, 0
        ER = re.compile(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$')
        PR = re.compile(r'^[\+\d][\d\s\-\.\(\)]{6,20}$')
        ecols = self.email_columns or [col for col in df.columns if any(kw in col.lower() for kw in ["email","mail"])]
        for col in ecols:
            if col not in df.columns: continue
            s = df[col].dropna().astype(str); inv = s[~s.str.match(ER)].count()
            c += len(s); v += inv
            if inv > 0:
                pct = inv / len(s) * 100
                r.issues.append({"dimension":"validity",
                    "severity":"high" if pct > 20 else "medium",
                    "column":col,"message":f"{inv:,} emails invalides ({pct:.1f}%)"})
        pcols = self.phone_columns or [col for col in df.columns if any(kw in col.lower() for kw in ["phone","tel","mobile","gsm"])]
        for col in pcols:
            if col not in df.columns: continue
            s = df[col].dropna().astype(str); inv = s[~s.str.match(PR)].count()
            c += len(s); v += inv
            if inv > 0:
                r.issues.append({"dimension":"validity","severity":"medium",
                    "column":col,"message":f"{inv:,} numéros invalides"})
        return 90.0 if c == 0 else round(max(0, (1 - v / c) * 100), 1)

    def _correlation(self, df, r):
        v, c = 0, 0
        for rule in self.correlation_rules:
            ca, cb, op = rule.get("col_a"), rule.get("col_b"), rule.get("operator","<")
            if ca not in df.columns or cb not in df.columns: continue
            try:
                a = pd.to_numeric(df[ca], errors="coerce")
                b = pd.to_numeric(df[cb], errors="coerce")
                mask = a.notna() & b.notna(); c += mask.sum()
                viol = (a[mask] >= b[mask]).sum() if op == "<" else (a[mask] > b[mask]).sum()
                v += viol
                if viol > 0:
                    rule_name = rule.get("name","")
                    r.issues.append({"dimension":"correlation",
                        "severity":rule.get("severity","high"),
                        "column":f"{ca}/{cb}",
                        "message":f"{viol:,} violations de '{rule_name}'"})
            except: pass
        return 90.0 if c == 0 else round(max(0, (1 - v / c) * 100), 1)

    def _standardization(self, df, r):
        FN = {"n/a","na","null","none","-","--","unknown","inconnu","?","nan","nd"}
        v, c = 0, 0
        for col in df.select_dtypes(include=["object"]).columns:
            s = df[col].dropna().astype(str)
            if not len(s): continue
            c += len(s)
            sp = (s != s.str.strip()).sum(); v += sp
            if sp > 0:
                r.issues.append({"dimension":"standardization","severity":"low",
                    "column":col,"message":f"{sp:,} valeurs avec espaces superflus"})
            fn = s.str.lower().str.strip().isin(FN).sum(); v += fn
            if fn > 0:
                r.issues.append({"dimension":"standardization","severity":"medium",
                    "column":col,"message":f"{fn:,} faux nulls (N/A, null, -…)"})
            nu = s.nunique(); nl = s.str.lower().str.strip().nunique()
            if 2 <= nu <= 50 and nl < nu:
                v += (nu - nl) * 10
                r.issues.append({"dimension":"standardization","severity":"medium",
                    "column":col,"message":f"Casse inconsistante : {nu} variantes pour {nl} valeurs réelles"})
        return 90.0 if c == 0 else round(max(0, min(100, (1 - v / c) * 100)), 1)

    def _column_scores(self, df):
        out = []
        for col in df.columns:
            s = df[col]
            cp = round((1 - s.isnull().mean()) * 100, 1)
            uq = round(s.nunique() / max(len(s), 1) * 100, 1)
            overall = min(round(cp * 0.6 + min(uq * 1.5, 100) * 0.4, 1), 100)
            out.append(ColumnScore(name=col, completeness=cp, uniqueness=uq, overall=overall))
        return out


# ══════════════════════════════════════════════════════════════
# ENGINE PYSPARK (optionnel — si cluster disponible)
# ══════════════════════════════════════════════════════════════

class PySparkScorer:
    """
    Engine PySpark natif — pour clients avec Databricks.
    Identique à PandasScorer mais 100% distribué, 100M+ lignes.
    Instancié uniquement si SPARK_AVAILABLE = True.
    """

    WEIGHTS = {
        "completeness":0.20,"consistency":0.15,"validity":0.15,
        "uniqueness":0.12,"freshness":0.10,"distribution":0.08,
        "correlation":0.08,"volumetry":0.07,"standardization":0.05,
    }

    def __init__(self, spark, table_name="dataset", date_columns=None,
                 email_columns=None, phone_columns=None, custom_rules=None,
                 correlation_rules=None, freshness_threshold_hours=24):
        self.spark = spark
        self.table_name = table_name
        self.date_columns = date_columns or []
        self.email_columns = email_columns or []
        self.phone_columns = phone_columns or []
        self.custom_rules = custom_rules or []
        self.correlation_rules = correlation_rules or []
        self.freshness_threshold_hours = freshness_threshold_hours

    def score(self, df) -> TableScore:
        row_count = df.count()
        r = TableScore(
            table_name=self.table_name,
            row_count=row_count,
            col_count=len(df.columns),
            engine="pyspark",
            custom_rules=self.custom_rules,
        )
        r.completeness    = self._completeness(df, r, row_count)
        r.uniqueness      = self._uniqueness(df, r, row_count)
        r.freshness       = self._freshness(df, r)
        r.consistency     = self._consistency(df, r, row_count)
        r.distribution    = self._distribution(df, r)
        r.validity        = self._validity(df, r, row_count)
        r.correlation     = self._correlation(df, r, row_count)
        r.volumetry       = 80.0
        r.standardization = self._standardization(df, r, row_count)
        r.global_score    = round(sum(getattr(r, d) * w for d, w in self.WEIGHTS.items()), 1)
        r.columns         = self._column_scores(df, row_count)
        return r

    def _completeness(self, df, r, row_count):
        if row_count == 0: return 0.0
        null_counts = df.select([
            F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns
        ]).collect()[0].asDict()
        total_nulls = sum(null_counts.values())
        for col, nc in null_counts.items():
            pct = nc / row_count * 100
            if pct > 20:
                r.issues.append({"dimension":"completeness",
                    "severity":"high" if pct > 50 else "medium",
                    "column":col,"message":f"{pct:.1f}% de valeurs nulles ({nc:,}/{row_count:,})"})
        return round((1 - total_nulls / (row_count * len(df.columns))) * 100, 1)

    def _uniqueness(self, df, r, row_count):
        if row_count < 2: return 100.0
        dup = row_count - df.dropDuplicates().count()
        pct = dup / row_count * 100
        if pct > 5:
            r.issues.append({"dimension":"uniqueness",
                "severity":"high" if pct > 20 else "medium",
                "column":"all","message":f"{dup:,} lignes dupliquées ({pct:.1f}%)"})
        return round(max(0, 100 - pct * 2), 1)

    def _freshness(self, df, r):
        if not self.date_columns: return 75.0
        cols = [c for c in self.date_columns if c in df.columns]
        if not cols: return 75.0
        max_dates = df.select([F.max(F.to_timestamp(c)).alias(c) for c in cols]).collect()[0].asDict()
        scores = []; now = datetime.now()
        for col, max_date in max_dates.items():
            if not max_date: continue
            lag = (now - max_date).total_seconds() / 3600
            scores.append(max(0, 100 - (lag / self.freshness_threshold_hours) * 100))
            if lag > self.freshness_threshold_hours:
                r.issues.append({"dimension":"freshness",
                    "severity":"high" if lag > self.freshness_threshold_hours * 3 else "medium",
                    "column":col,"message":f"Dernière donnée il y a {lag:.0f}h"})
        return round(sum(scores) / len(scores), 1) if scores else 75.0

    def _consistency(self, df, r, row_count):
        v, c = 0, 0
        num_cols = [f.name for f in df.schema.fields if isinstance(f.dataType,
            (T.DoubleType, T.FloatType, T.IntegerType, T.LongType))]
        for col in num_cols:
            if any(kw in col.lower() for kw in ["price","prix","amount","montant","age","qty","quantity","stock"]):
                neg = df.filter(F.col(col) < 0).count(); v += neg; c += row_count
                if neg > 0:
                    r.issues.append({"dimension":"consistency","severity":"high",
                        "column":col,"message":f"{neg:,} valeurs négatives"})
        for rule in self.custom_rules:
            try:
                n = df.filter(~F.expr(rule["condition"])).count(); c += row_count; v += n
                if n > 0:
                    r.issues.append({"dimension":"consistency",
                        "severity":rule.get("severity","medium"),
                        "column":rule.get("column","custom"),
                        "message":f"Règle '{rule['name']}': {n:,} violations"})
            except: pass
        return 90.0 if c == 0 else round(max(0, (1 - v / c) * 100), 1)

    def _distribution(self, df, r):
        num_cols = [f.name for f in df.schema.fields if isinstance(f.dataType,
            (T.DoubleType, T.FloatType, T.IntegerType, T.LongType))]
        if not num_cols: return 90.0
        try:
            quantiles = df.approxQuantile(num_cols, [0.25, 0.75], 0.01)
        except: return 90.0
        ratios = []
        for col, (q1, q3) in zip(num_cols, quantiles):
            if q1 is None or q3 is None: continue
            iqr = q3 - q1
            if iqr == 0: continue
            out = df.filter((F.col(col) < q1 - 3*iqr) | (F.col(col) > q3 + 3*iqr)).count()
            total = df.filter(F.col(col).isNotNull()).count()
            if total == 0: continue
            ratio = out / total; ratios.append(ratio)
            if ratio > 0.05:
                r.issues.append({"dimension":"distribution","severity":"medium",
                    "column":col,"message":f"{out:,} outliers extrêmes ({ratio*100:.1f}%)"})
        return 90.0 if not ratios else round(max(0, 100 - (sum(ratios)/len(ratios)) * 500), 1)

    def _validity(self, df, r, row_count):
        v, c = 0, 0
        ER = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
        PR = r'^\+?[\d][\d\s\-\.\(\)]{6,20}$'
        ecols = self.email_columns or [col for col in df.columns if any(kw in col.lower() for kw in ["email","mail"])]
        for col in ecols:
            if col not in df.columns: continue
            nn = df.filter(F.col(col).isNotNull()); total = nn.count()
            inv = nn.filter(~F.col(col).rlike(ER)).count()
            c += total; v += inv
            if inv > 0:
                pct = inv / total * 100
                r.issues.append({"dimension":"validity",
                    "severity":"high" if pct > 20 else "medium",
                    "column":col,"message":f"{inv:,} emails invalides ({pct:.1f}%)"})
        pcols = self.phone_columns or [col for col in df.columns if any(kw in col.lower() for kw in ["phone","tel","mobile","gsm"])]
        for col in pcols:
            if col not in df.columns: continue
            nn = df.filter(F.col(col).isNotNull()); total = nn.count()
            inv = nn.filter(~F.col(col).cast("string").rlike(PR)).count()
            c += total; v += inv
            if inv > 0:
                r.issues.append({"dimension":"validity","severity":"medium",
                    "column":col,"message":f"{inv:,} numéros invalides"})
        return 90.0 if c == 0 else round(max(0, (1 - v / c) * 100), 1)

    def _correlation(self, df, r, row_count):
        v, c = 0, 0
        for rule in self.correlation_rules:
            ca, cb, op = rule.get("col_a"), rule.get("col_b"), rule.get("operator","<")
            if ca not in df.columns or cb not in df.columns: continue
            try:
                both = df.filter(F.col(ca).isNotNull() & F.col(cb).isNotNull())
                total = both.count(); c += total
                cond = F.col(ca) >= F.col(cb) if op == "<" else F.col(ca) > F.col(cb)
                viol = both.filter(cond).count(); v += viol
                if viol > 0:
                    rule_name = rule.get("name","")
                    r.issues.append({"dimension":"correlation",
                        "severity":rule.get("severity","high"),
                        "column":f"{ca}/{cb}",
                        "message":f"{viol:,} violations de '{rule_name}'"})
            except: pass
        return 90.0 if c == 0 else round(max(0, (1 - v / c) * 100), 1)

    def _standardization(self, df, r, row_count):
        FN = ["n/a","na","null","none","-","--","unknown","inconnu","?","nan","nd"]
        str_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, T.StringType)]
        if not str_cols: return 90.0
        v, c = 0, 0
        for col in str_cols:
            nn = df.filter(F.col(col).isNotNull()); total = nn.count()
            if total == 0: continue
            c += total
            sp = nn.filter(F.col(col) != F.trim(F.col(col))).count(); v += sp
            if sp > 0:
                r.issues.append({"dimension":"standardization","severity":"low",
                    "column":col,"message":f"{sp:,} valeurs avec espaces superflus"})
            fn = nn.filter(F.lower(F.trim(F.col(col))).isin(FN)).count(); v += fn
            if fn > 0:
                r.issues.append({"dimension":"standardization","severity":"medium",
                    "column":col,"message":f"{fn:,} faux nulls (N/A, null, -…)"})
            nu = nn.select(F.col(col)).distinct().count()
            nl = nn.select(F.lower(F.trim(F.col(col))).alias(col)).distinct().count()
            if 2 <= nu <= 50 and nl < nu:
                v += (nu - nl) * 10
                r.issues.append({"dimension":"standardization","severity":"medium",
                    "column":col,"message":f"Casse inconsistante : {nu} variantes pour {nl} valeurs réelles"})
        return 90.0 if c == 0 else round(max(0, min(100, (1 - v / c) * 100)), 1)

    def _column_scores(self, df, row_count):
        out = []
        null_counts = df.select([
            F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns
        ]).collect()[0].asDict()
        for col in df.columns:
            nc = null_counts.get(col, 0)
            cp = round((1 - nc / max(row_count, 1)) * 100, 1)
            nd = df.agg(F.approx_count_distinct(F.col(col)).alias("n")).collect()[0]["n"]
            uq = round(nd / max(row_count, 1) * 100, 1)
            overall = min(round(cp * 0.6 + min(uq * 1.5, 100) * 0.4, 1), 100)
            out.append(ColumnScore(name=col, completeness=cp, uniqueness=uq, overall=overall))
        return out


# ══════════════════════════════════════════════════════════════
# FACADE — Point d'entrée unique
# ══════════════════════════════════════════════════════════════

def run_scoring(
    df,
    table_name:               str  = "dataset",
    custom_rules:             list = None,
    freshness_threshold_hours: int = 24,
    spark=None,
) -> TableScore:
    """
    Point d'entrée unique.
    - df peut être un pandas DataFrame OU un Spark DataFrame
    - Si spark est fourni ET df est Spark → PySpark engine
    - Sinon → Pandas engine
    Auto-détecte les colonnes email/phone/date/corrélation.
    """
    custom_rules = custom_rules or []

    # Convertir Spark → pandas si pas de spark fourni
    if SPARK_AVAILABLE and hasattr(df, "toPandas") and spark is None:
        df = df.toPandas()

    # Auto-détection sur pandas
    pdf = df if isinstance(df, pd.DataFrame) else df.toPandas()
    detected = ColumnAutoDetector().detect(pdf)

    common_args = dict(
        table_name=table_name,
        date_columns=detected["date_columns"],
        email_columns=detected["email_columns"],
        phone_columns=detected["phone_columns"],
        correlation_rules=detected["correlation_rules"],
        custom_rules=custom_rules,
        freshness_threshold_hours=freshness_threshold_hours,
    )

    # Choisir l'engine
    if spark is not None and SPARK_AVAILABLE and not isinstance(df, pd.DataFrame):
        print(f"[DQ] Engine : PySpark (distribué)")
        scorer = PySparkScorer(spark=spark, **common_args)
        return scorer.score(df)
    else:
        print(f"[DQ] Engine : Pandas")
        scorer = PandasScorer(**common_args)
        pdf_df = df if isinstance(df, pd.DataFrame) else df.toPandas()
        return scorer.score(pdf_df)


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DataQuality Agent",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════
# STYLES
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Cabinet+Grotesk:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:       #FAFAF8;
  --surface:  #FFFFFF;
  --surface2: #F5F4F0;
  --border:   #E8E6E0;
  --border2:  #D4D0C8;
  --accent:   #3730A3;
  --accent2:  #4F46E5;
  --accent-l: #EEF2FF;
  --ok:       #059669;
  --ok-l:     #ECFDF5;
  --warn:     #D97706;
  --warn-l:   #FFFBEB;
  --danger:   #DC2626;
  --danger-l: #FEF2F2;
  --text:     #1C1917;
  --text2:    #44403C;
  --muted:    #78716C;
  --dim:      #A8A29E;
}

*,*::before,*::after { box-sizing:border-box; margin:0; }
html,body,[class*="css"] {
  font-family:'Cabinet Grotesk',sans-serif;
  background:var(--bg);
  color:var(--text);
}
.stApp { background:var(--bg); }
#MainMenu,footer,header { visibility:hidden; }
section[data-testid="stSidebar"] { display:none; }

::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border2); border-radius:3px; }

/* ── NAV ── */
.topnav {
  display:flex; align-items:center; justify-content:space-between;
  padding:14px 40px;
  border-bottom:1px solid var(--border);
  background:rgba(250,250,248,0.92);
  backdrop-filter:blur(12px);
  position:sticky; top:0; z-index:100;
}
.brand { display:flex; align-items:center; gap:10px; }
.brand-icon {
  width:34px; height:34px;
  background:var(--accent);
  border-radius:9px;
  display:flex; align-items:center; justify-content:center;
  font-size:1rem; color:white;
  box-shadow:0 2px 8px rgba(55,48,163,0.25);
}
.brand-name {
  font-family:'Cabinet Grotesk',sans-serif;
  font-weight:700; font-size:1rem; color:var(--text);
  letter-spacing:-0.2px;
}
.brand-tag {
  font-size:0.6rem; color:var(--muted);
  letter-spacing:1.5px; text-transform:uppercase;
}
.nav-right { display:flex; align-items:center; gap:10px; }
.user-pill {
  background:var(--surface2); border:1px solid var(--border);
  border-radius:20px; padding:5px 14px;
  font-size:0.75rem; color:var(--muted); font-weight:500;
}
.engine-pill {
  background:var(--accent-l); border:1px solid rgba(55,48,163,0.2);
  border-radius:20px; padding:4px 12px;
  font-size:0.65rem; color:var(--accent2);
  font-family:'JetBrains Mono',monospace; font-weight:500;
}

/* ── STEPS ── */
.steps { display:flex; align-items:center; }
.step {
  display:flex; align-items:center; gap:7px;
  padding:5px 14px; border-radius:20px;
  font-size:0.75rem; font-weight:600; color:var(--dim);
}
.step.active { background:var(--accent-l); color:var(--accent2); }
.step.done   { color:var(--ok); }
.step-num {
  width:20px; height:20px; border-radius:50%;
  background:var(--border); color:var(--muted);
  display:flex; align-items:center; justify-content:center;
  font-size:0.6rem; font-weight:700;
}
.step.active .step-num { background:var(--accent2); color:white; }
.step.done   .step-num { background:var(--ok); color:white; }
.step-sep { color:var(--dim); font-size:0.65rem; padding:0 2px; }

/* ── LOGIN ── */
.login-wrap { max-width:420px; margin:72px auto 0; padding:0 20px; }
.login-logo { text-align:center; margin-bottom:36px; }
.login-logo-icon {
  width:56px; height:56px; background:var(--accent); border-radius:14px;
  display:flex; align-items:center; justify-content:center;
  font-size:1.8rem; margin:0 auto 14px;
  box-shadow:0 8px 24px rgba(55,48,163,0.3);
}
.login-title {
  font-family:'Instrument Serif',serif;
  font-size:1.9rem; font-weight:400; color:var(--text); letter-spacing:-0.5px;
}
.login-sub { font-size:0.85rem; color:var(--muted); margin-top:6px; }
.login-card {
  background:var(--surface); border:1px solid var(--border);
  border-radius:20px; padding:32px;
  box-shadow:0 4px 24px rgba(0,0,0,0.06);
}

/* ── CARD LABEL ── */
.card-label {
  font-family:'JetBrains Mono',monospace;
  font-size:0.62rem; color:var(--dim);
  text-transform:uppercase; letter-spacing:2px;
  margin-bottom:16px;
  display:flex; align-items:center; gap:10px;
}
.card-label::after { content:''; flex:1; height:1px; background:var(--border); }

/* ── HERO ── */
.hero {
  padding:52px 32px 36px;
  max-width:700px; margin:0 auto; text-align:center;
}
.hero-eyebrow {
  font-family:'JetBrains Mono',monospace;
  font-size:0.68rem; color:var(--accent2);
  letter-spacing:3px; text-transform:uppercase; margin-bottom:20px;
  display:flex; align-items:center; justify-content:center; gap:10px;
}
.hero-eyebrow::before,.hero-eyebrow::after {
  content:''; width:32px; height:1px; background:var(--accent2); opacity:.3;
}
.hero-title {
  font-family:'Instrument Serif',serif;
  font-size:3.4rem; font-weight:400; line-height:1.1;
  letter-spacing:-1px; color:var(--text); margin-bottom:14px;
}
.hero-title span { color:var(--accent2); font-style:italic; }
.hero-sub {
  font-size:1rem; color:var(--muted);
  font-weight:400; line-height:1.65; margin-bottom:36px;
}
.hero-stats { display:flex; justify-content:center; gap:40px; margin-bottom:44px; }
.hero-stat-n {
  font-family:'Instrument Serif',serif;
  font-size:2rem; font-weight:400; color:var(--text);
}
.hero-stat-l {
  font-size:0.65rem; color:var(--muted);
  text-transform:uppercase; letter-spacing:1px; margin-top:1px;
}

/* ── SOURCE CARDS ── */
.src-card {
  background:var(--surface); border:1.5px solid var(--border);
  border-radius:14px; padding:20px 14px;
  text-align:center; transition:all .18s;
  box-shadow:0 1px 4px rgba(0,0,0,0.04);
  cursor:pointer; position:relative;
}
.src-card:hover {
  border-color:var(--accent2); transform:translateY(-2px);
  box-shadow:0 6px 20px rgba(55,48,163,0.1);
}
.src-card.selected {
  border-color:var(--accent2);
  background:var(--accent-l);
  box-shadow:0 4px 16px rgba(55,48,163,0.12);
}
.src-card.selected::after {
  content:"✓";
  position:absolute; top:8px; right:10px;
  width:18px; height:18px; border-radius:50%;
  background:var(--accent2); color:white;
  font-size:0.6rem; font-weight:700;
  display:flex; align-items:center; justify-content:center;
  line-height:18px;
}
.src-icon { font-size:1.8rem; margin-bottom:8px; }
.src-name {
  font-family:'Cabinet Grotesk',sans-serif;
  font-size:0.83rem; font-weight:700; color:var(--text);
}
.src-desc { font-size:0.67rem; color:var(--muted); margin-top:2px; }

/* SRC CARD BUTTON — bouton qui ressemble à une carte */
div[data-testid="stButton"].src-btn > button {
  background:var(--surface) !important;
  border:1.5px solid var(--border) !important;
  border-radius:14px !important;
  padding:20px 14px !important;
  width:100% !important;
  text-align:center !important;
  transition:all .18s !important;
  box-shadow:0 1px 4px rgba(0,0,0,0.04) !important;
  color:var(--text) !important;
  font-family:"Cabinet Grotesk",sans-serif !important;
  height:auto !important;
  white-space:normal !important;
  line-height:1.4 !important;
}
div[data-testid="stButton"].src-btn > button:hover {
  border-color:var(--accent2) !important;
  transform:translateY(-2px) !important;
  box-shadow:0 6px 20px rgba(55,48,163,0.1) !important;
}

/* HELP BOX */
.help-box {
  background:linear-gradient(135deg,#EFF6FF,#EEF2FF);
  border:1.5px solid #BFDBFE; border-radius:12px;
  padding:16px 18px; margin-bottom:20px;
}
.help-box-title {
  font-family:'Cabinet Grotesk',sans-serif;
  font-weight:700; font-size:0.85rem; color:#1E40AF;
  display:flex; align-items:center; gap:7px; margin-bottom:8px;
}
.help-box-body { font-size:0.78rem; color:#3730A3; line-height:1.6; }
.help-box-body code {
  background:rgba(55,48,163,0.1); padding:1px 5px;
  border-radius:3px; font-family:'JetBrains Mono',monospace;
  font-size:0.72rem;
}
.help-example {
  background:white; border:1px solid #BFDBFE;
  border-radius:8px; padding:10px 14px; margin-top:10px;
  font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#1E40AF;
}

/* ── DATABRICKS BLOCK ── */
.dbx-card {
  background:linear-gradient(135deg, #FFF7ED 0%, #FEF3C7 100%);
  border:1.5px solid #FCD34D;
  border-radius:14px; padding:20px 22px; margin-top:20px;
}
.dbx-header {
  display:flex; align-items:center; gap:10px; margin-bottom:14px;
}
.dbx-icon {
  width:32px; height:32px; background:#FF3621;
  border-radius:8px; display:flex; align-items:center;
  justify-content:center; font-size:1rem; color:white;
}
.dbx-title {
  font-family:'Cabinet Grotesk',sans-serif;
  font-weight:700; font-size:0.95rem; color:#92400E;
}
.dbx-sub { font-size:0.72rem; color:#B45309; margin-top:1px; }
.dbx-badge-on {
  display:inline-flex; align-items:center; gap:5px;
  background:#ECFDF5; border:1px solid #6EE7B7;
  border-radius:20px; padding:3px 10px;
  font-size:0.65rem; font-weight:600; color:#065F46;
  font-family:'JetBrains Mono',monospace;
}
.dbx-badge-off {
  display:inline-flex; align-items:center; gap:5px;
  background:#F5F5F4; border:1px solid var(--border);
  border-radius:20px; padding:3px 10px;
  font-size:0.65rem; font-weight:600; color:var(--muted);
  font-family:'JetBrains Mono',monospace;
}

/* ── SCORE ── */
.score-ring { text-align:center; padding:32px 20px; }
.score-number {
  font-family:'Instrument Serif',serif;
  font-size:5.5rem; font-weight:400; line-height:1; letter-spacing:-3px;
}
.score-denom { font-size:1.4rem; color:var(--muted); font-weight:400; }
.score-badge {
  display:inline-flex; align-items:center; gap:6px;
  padding:5px 16px; border-radius:20px;
  font-size:0.7rem; font-weight:700; letter-spacing:1.5px;
  text-transform:uppercase; margin-top:14px; border:1.5px solid;
}
.score-meta {
  font-family:'JetBrains Mono',monospace;
  font-size:0.65rem; color:var(--dim); margin-top:10px;
}

/* ── DIM TILES ── */
.dim-tile {
  background:var(--surface); border:1.5px solid var(--border);
  border-radius:12px; padding:14px;
  position:relative; overflow:hidden;
  box-shadow:0 1px 4px rgba(0,0,0,0.04);
}
.dim-tile::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; }
.dim-tile.ok::before    { background:var(--ok); }
.dim-tile.warn::before  { background:var(--warn); }
.dim-tile.danger::before { background:var(--danger); }
.dim-score {
  font-family:'Instrument Serif',serif;
  font-size:1.7rem; font-weight:400; line-height:1;
}
.dim-name {
  font-size:0.62rem; font-weight:700; color:var(--muted);
  text-transform:uppercase; letter-spacing:0.8px; margin-top:4px;
}
.dim-weight { font-size:0.56rem; color:var(--dim); margin-top:1px; }
.dim-bar {
  height:3px; background:var(--border);
  border-radius:2px; margin-top:8px; overflow:hidden;
}
.dim-bar-fill { height:100%; border-radius:2px; }

/* ── ISSUES ── */
.issue {
  display:flex; align-items:flex-start; gap:10px;
  padding:11px 14px; border-radius:10px;
  margin-bottom:5px; border-left:3px solid;
}
.issue.high   { background:var(--danger-l); border-color:var(--danger); }
.issue.medium { background:var(--warn-l);   border-color:var(--warn); }
.issue.low    { background:var(--ok-l);     border-color:var(--ok); }
.issue-sev {
  font-family:'JetBrains Mono',monospace;
  font-size:0.56rem; font-weight:700;
  padding:2px 6px; border-radius:4px;
  white-space:nowrap; text-transform:uppercase; flex-shrink:0;
}
.sev-high   { background:rgba(220,38,38,.12); color:#B91C1C; }
.sev-medium { background:rgba(217,119,6,.12); color:#92400E; }
.sev-low    { background:rgba(5,150,105,.12); color:#065F46; }
.issue-dim  {
  font-family:'JetBrains Mono',monospace;
  font-size:0.62rem; color:var(--accent2); margin-bottom:2px;
}
.issue-msg { font-size:0.8rem; color:var(--text2); }
.issue-col { font-family:'JetBrains Mono',monospace; color:var(--accent2); }

/* ── RULE BUILDER ── */
.rule-item {
  background:var(--surface2); border:1px solid var(--border);
  border-radius:10px; padding:12px 14px; margin-bottom:6px;
}
.rule-name-t { font-weight:700; font-size:0.82rem; color:var(--text); }
.rule-cond-t {
  font-family:'JetBrains Mono',monospace;
  font-size:0.66rem; color:var(--accent2); margin-top:3px;
}

/* ── BUTTONS ── */
.stButton>button {
  font-family:'Cabinet Grotesk',sans-serif !important;
  font-weight:700 !important; border-radius:10px !important;
  border:none !important; padding:10px 22px !important;
  transition:all .15s !important;
}
.stButton>button[kind="primary"] {
  background:var(--accent2) !important; color:white !important;
  box-shadow:0 2px 8px rgba(79,70,229,0.2) !important;
}
.stButton>button[kind="primary"]:hover {
  background:var(--accent) !important; transform:translateY(-1px) !important;
  box-shadow:0 6px 18px rgba(55,48,163,0.28) !important;
}
.stButton>button[kind="secondary"] {
  background:var(--surface) !important; color:var(--text) !important;
  border:1.5px solid var(--border) !important;
}
.stButton>button[kind="secondary"]:hover {
  border-color:var(--border2) !important;
  box-shadow:0 2px 8px rgba(0,0,0,0.06) !important;
}

/* ── INPUTS ── */
.stTextInput>div>div>input,
.stTextArea>div>div>textarea,
.stSelectbox>div>div,
.stNumberInput>div>div>input {
  background:var(--surface) !important;
  border:1.5px solid var(--border) !important;
  border-radius:9px !important; color:var(--text) !important;
  font-family:'Cabinet Grotesk',sans-serif !important;
  font-size:0.88rem !important;
}
.stTextInput>div>div>input:focus,
.stTextArea>div>div>textarea:focus {
  border-color:var(--accent2) !important;
  box-shadow:0 0 0 3px rgba(79,70,229,0.1) !important;
}
label { color:var(--muted) !important; font-size:0.78rem !important; font-weight:500 !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
  background:var(--surface2); border-radius:10px;
  padding:4px; gap:2px; border:1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  background:transparent !important; color:var(--muted) !important;
  border-radius:8px !important;
  font-family:'Cabinet Grotesk',sans-serif !important;
  font-weight:600 !important; font-size:0.8rem !important;
}
.stTabs [aria-selected="true"] {
  background:var(--surface) !important; color:var(--accent2) !important;
  box-shadow:0 1px 4px rgba(0,0,0,0.08) !important;
}

/* ── ALERTS ── */
.alert { padding:12px 16px; border-radius:10px; margin-bottom:12px; font-size:0.82rem; font-weight:500; }
.alert-ok   { background:var(--ok-l);     border:1px solid #A7F3D0; color:var(--ok); }
.alert-warn { background:var(--warn-l);   border:1px solid #FDE68A; color:var(--warn); }
.alert-err  { background:var(--danger-l); border:1px solid #FECACA; color:var(--danger); }
.alert-info { background:var(--accent-l); border:1px solid #C7D2FE; color:var(--accent2); }

/* ── METRICS ── */
.metrics-row { display:flex; gap:10px; margin-bottom:14px; }
.metric {
  flex:1; background:var(--surface); border:1.5px solid var(--border);
  border-radius:12px; padding:16px;
  box-shadow:0 1px 4px rgba(0,0,0,0.04);
}
.metric-val {
  font-family:'Instrument Serif',serif;
  font-size:1.7rem; font-weight:400; color:var(--text);
}
.metric-lbl {
  font-size:0.65rem; color:var(--muted);
  text-transform:uppercase; letter-spacing:1px; margin-top:2px; font-weight:600;
}

/* ── SECTION ── */
.sec-title {
  font-family:'Instrument Serif',serif;
  font-size:1.4rem; font-weight:400; color:var(--text);
  letter-spacing:-0.3px; margin-bottom:4px;
}
.sec-sub { font-size:0.82rem; color:var(--muted); margin-bottom:20px; }

/* ── MISC ── */
hr { border-color:var(--border) !important; margin:28px 0 !important; }
[data-testid="stFileUploader"] {
  background:var(--surface) !important;
  border:2px dashed var(--border2) !important;
  border-radius:12px !important;
}
.stDataFrame { border:1px solid var(--border) !important; border-radius:10px !important; }
div[data-testid="stExpander"] {
  background:var(--surface) !important;
  border:1px solid var(--border) !important;
  border-radius:12px !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PDF GENERATOR
# ══════════════════════════════════════════════════════════════

def clean(t):
    return (str(t).replace("—","-").replace("–","-")
        .replace("é","e").replace("è","e").replace("ê","e").replace("ë","e")
        .replace("à","a").replace("â","a").replace("ù","u").replace("û","u")
        .replace("î","i").replace("ï","i").replace("ô","o").replace("ç","c")
        .replace("É","E").replace("È","E").replace("'","'").replace("…","..."))

def generate_pdf(result: TableScore) -> bytes:
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    C_PRI=(37,99,235); C_WHT=(255,255,255); C_GRY=(107,114,128)
    C_OK=(16,185,129); C_WRN=(245,158,11); C_ERR=(239,68,68)
    C_DRK=(14,20,32);  C_LGT=(26,34,53);   C_BG=(8,11,18)

    def sc(s): return C_OK if s>=80 else (C_WRN if s>=60 else C_ERR)
    def sl(s): return "BON" if s>=80 else ("MOYEN" if s>=60 else "CRITIQUE")

    class PDF(FPDF):
        def header(self):
            self.set_fill_color(*C_BG);  self.rect(0,0,210,297,"F")
            self.set_fill_color(*C_PRI); self.rect(0,0,210,30,"F")
            self.set_text_color(*C_WHT); self.set_font("Helvetica","B",17)
            self.set_xy(12,8); self.cell(0,10,"DataQuality Agent")
            self.set_font("Helvetica","",8); self.set_xy(12,21)
            eng = result.engine.upper()
            self.cell(0,6,f"Rapport genere le {datetime.now().strftime('%d/%m/%Y a %H:%M')} | Engine : {eng} | One-Shot")
            self.ln(34)
        def footer(self):
            self.set_y(-14); self.set_font("Helvetica","I",8)
            self.set_text_color(*C_GRY)
            self.cell(0,8,f"DataQuality Agent v3 | Aucune donnee stockee | Page {self.page_no()}",align="C")

    pdf=PDF(); pdf.add_page(); pdf.set_auto_page_break(True,18)

    # Score global
    score=result.global_score; color=sc(score)
    pdf.set_fill_color(*C_DRK); pdf.rect(12,34,186,46,"F")
    pdf.set_fill_color(*color); pdf.rect(12,34,4,46,"F")
    pdf.set_text_color(*color); pdf.set_font("Helvetica","B",46)
    pdf.set_xy(22,36); pdf.cell(70,18,f"{score}")
    pdf.set_font("Helvetica","",15); pdf.set_text_color(*C_GRY)
    pdf.set_xy(72,49); pdf.cell(0,8,"/ 100")
    pdf.set_font("Helvetica","B",11); pdf.set_text_color(*color)
    pdf.set_xy(120,40); pdf.cell(72,8,f"Score {sl(score)}",align="R")
    pdf.set_font("Helvetica","",8); pdf.set_text_color(*C_GRY)
    pdf.set_xy(120,50); pdf.cell(72,6,clean(f"{result.row_count:,} lignes  |  {result.col_count} colonnes"),align="R")
    pdf.set_xy(120,57); pdf.cell(72,6,clean(result.table_name),align="R")
    pdf.ln(8)

    # Dimensions
    pdf.set_font("Helvetica","B",8); pdf.set_text_color(*C_GRY)
    pdf.set_xy(12,pdf.get_y()+4); pdf.cell(0,6,"SCORES PAR DIMENSION",ln=True)
    pdf.set_fill_color(*C_LGT); pdf.rect(12,pdf.get_y(),186,1,"F"); pdf.ln(2)

    dims=[("Completude",result.completeness,"20%"),("Coherence",result.consistency,"15%"),
          ("Validite",result.validity,"15%"),("Unicite",result.uniqueness,"12%"),
          ("Fraicheur",result.freshness,"10%"),("Distribution",result.distribution,"8%"),
          ("Correlation",result.correlation,"8%"),("Volumetrie",result.volumetry,"7%"),
          ("Standard.",result.standardization,"5%")]

    for i,(name,sd,w) in enumerate(dims):
        c=sc(sd); y=pdf.get_y()
        pdf.set_fill_color(*(C_DRK if i%2==0 else (18,24,38))); pdf.rect(12,y,186,10,"F")
        pdf.set_fill_color(*c); pdf.rect(12,y,3,10,"F")
        pdf.set_text_color(*C_GRY); pdf.set_font("Helvetica","",8)
        pdf.set_xy(18,y+2); pdf.cell(50,6,name)
        bw=88*sd/100
        pdf.set_fill_color(*(22,30,50)); pdf.rect(70,y+3.5,88,3,"F")
        pdf.set_fill_color(*c);          pdf.rect(70,y+3.5,bw,3,"F")
        pdf.set_text_color(*c); pdf.set_font("Helvetica","B",8)
        pdf.set_xy(162,y+2); pdf.cell(18,6,f"{sd}",align="R")
        pdf.set_text_color(*(40,55,80)); pdf.set_font("Helvetica","",7)
        pdf.set_xy(182,y+2); pdf.cell(14,6,w,align="R")
        pdf.ln(10)

    pdf.ln(4)

    # Règles custom
    if result.custom_rules:
        pdf.set_font("Helvetica","B",8); pdf.set_text_color(*C_GRY)
        pdf.cell(0,6,f"REGLES METIER ({len(result.custom_rules)})",ln=True)
        pdf.set_fill_color(*C_LGT); pdf.rect(12,pdf.get_y(),186,1,"F"); pdf.ln(2)
        for rule in result.custom_rules:
            y=pdf.get_y()
            pdf.set_fill_color(*C_DRK); pdf.rect(12,y,186,9,"F")
            sev_c={"high":C_ERR,"medium":C_WRN,"low":C_OK}.get(rule.get("severity","medium"),C_GRY)
            pdf.set_fill_color(*sev_c); pdf.rect(12,y,3,9,"F")
            pdf.set_text_color(*(100,140,200)); pdf.set_font("Helvetica","B",7)
            pdf.set_xy(18,y+1.5); pdf.cell(58,6,clean(rule.get("name","")[:32]))
            pdf.set_text_color(*C_GRY); pdf.set_font("Helvetica","",7)
            pdf.set_xy(78,y+1.5); pdf.cell(118,6,clean(rule.get("condition","")[:55]))
            pdf.ln(9)
        pdf.ln(3)

    # Problèmes
    if result.issues:
        pdf.set_font("Helvetica","B",8); pdf.set_text_color(*C_GRY)
        pdf.cell(0,6,f"PROBLEMES DETECTES  ({len(result.issues)})",ln=True)
        pdf.set_fill_color(*C_LGT); pdf.rect(12,pdf.get_y(),186,1,"F"); pdf.ln(2)
        sc_map={"high":C_ERR,"medium":C_WRN,"low":C_OK}
        for iss in result.issues:
            c=sc_map.get(iss.get("severity","medium"),C_GRY); y=pdf.get_y()
            pdf.set_fill_color(*C_DRK); pdf.rect(12,y,186,10,"F")
            pdf.set_fill_color(*c); pdf.rect(12,y,3,10,"F")
            pdf.set_text_color(*c); pdf.set_font("Helvetica","B",6)
            pdf.set_xy(18,y+2); pdf.cell(18,6,iss.get("severity","").upper())
            pdf.set_text_color(*(80,120,180)); pdf.set_font("Helvetica","B",7)
            pdf.set_xy(38,y+2); pdf.cell(36,6,clean(str(iss.get("column",""))[:20]))
            pdf.set_text_color(*C_GRY); pdf.set_font("Helvetica","",7)
            pdf.set_xy(76,y+2); pdf.multi_cell(120,6,clean(str(iss.get("message",""))[:88]))
            if pdf.get_y()<y+10: pdf.set_y(y+10)

    # Colonnes
    pdf.ln(3)
    pdf.set_font("Helvetica","B",8); pdf.set_text_color(*C_GRY)
    pdf.cell(0,6,"QUALITE PAR COLONNE",ln=True)
    pdf.set_fill_color(*C_LGT); pdf.rect(12,pdf.get_y(),186,1,"F"); pdf.ln(2)
    pdf.set_fill_color(*C_PRI); pdf.set_text_color(*C_WHT); pdf.set_font("Helvetica","B",7)
    for h,w in [("Colonne",82),("Completude",30),("Unicite",30),("Score",32)]:
        pdf.cell(w,7,h,fill=True)
    pdf.ln(7)
    for i,col in enumerate(sorted(result.columns,key=lambda x:x.overall)[:14]):
        bg=C_DRK if i%2==0 else (18,24,38); c=sc(col.overall)
        pdf.set_fill_color(*bg); pdf.set_text_color(*C_GRY); pdf.set_font("Helvetica","",7)
        pdf.cell(82,6,clean(col.name[:38]),fill=True)
        pdf.cell(30,6,f"{col.completeness}%",fill=True)
        pdf.cell(30,6,f"{col.uniqueness}%",fill=True)
        pdf.set_text_color(*c); pdf.set_font("Helvetica","B",7)
        pdf.cell(32,6,f"{col.overall}",fill=True); pdf.ln(6)

    pdf.ln(5)
    pdf.set_font("Helvetica","I",7); pdf.set_text_color(*(25,35,55))
    pdf.cell(0,5,"Rapport genere en memoire. Aucune donnee client n'a ete stockee ou transmise. | DataQuality Agent v3",align="C")
    return bytes(pdf.output())


# ══════════════════════════════════════════════════════════════
# HELPERS UI
# ══════════════════════════════════════════════════════════════

def sc_cls(s): return "ok" if s>=80 else ("warn" if s>=60 else "danger")
def sc_hex(s): return "#10B981" if s>=80 else ("#F59E0B" if s>=60 else "#EF4444")
def sl(s):     return "BON" if s>=80 else ("MOYEN" if s>=60 else "CRITIQUE")
def se(s):     return "✅" if s>=80 else ("⚠️" if s>=60 else "🔴")

def render_issues(issues):
    if not issues:
        st.markdown('<div style="color:var(--muted);padding:24px;text-align:center;font-size:0.83rem;">Aucun problème dans cette catégorie 🎉</div>', unsafe_allow_html=True)
        return
    for i in issues:
        sev = i.get("severity","medium")
        st.markdown(f"""
        <div class="issue {sev}">
          <span class="issue-sev sev-{sev}">{sev}</span>
          <div>
            <div class="issue-dim">{i.get('dimension','').upper()} · <span class="issue-col">{i.get('column','')}</span></div>
            <div class="issue-msg">{i.get('message','')}</div>
          </div>
        </div>""", unsafe_allow_html=True)

def radar_chart(r):
    dims=["Complétude","Unicité","Fraîcheur","Cohérence","Distrib.","Validité","Corrél.","Volume","Standard."]
    vals=[r.completeness,r.uniqueness,r.freshness,r.consistency,r.distribution,
          r.validity,r.correlation,r.volumetry,r.standardization]
    fig=go.Figure(go.Scatterpolar(r=vals+[vals[0]],theta=dims+[dims[0]],fill="toself",
        fillcolor="rgba(59,130,246,.08)",line=dict(color="#3B82F6",width=2),marker=dict(size=4,color="#3B82F6")))
    fig.update_layout(polar=dict(bgcolor="rgba(14,20,32,.8)",
        radialaxis=dict(visible=True,range=[0,100],gridcolor="#1A2235",tickfont=dict(color="#334155",size=8)),
        angularaxis=dict(tickfont=dict(color="#64748B",size=9),gridcolor="#1A2235")),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,height=300,margin=dict(l=42,r=42,t=24,b=24))
    return fig

def bar_chart(r):
    data=[{"Colonne":c.name[:22],"Score":c.overall} for c in sorted(r.columns,key=lambda x:x.overall)[:14]]
    fig=px.bar(pd.DataFrame(data),x="Score",y="Colonne",orientation="h",
        color="Score",color_continuous_scale=["#EF4444","#F59E0B","#10B981"],range_color=[0,100],text="Score")
    fig.update_layout(height=340,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,margin=dict(l=0,r=14,t=6,b=6),
        yaxis=dict(tickfont=dict(color="#64748B",size=9)),
        xaxis=dict(gridcolor="#1A2235",tickfont=dict(color="#475569",size=9),range=[0,115]))
    fig.update_traces(textposition="outside",textfont=dict(color="#94A3B8",size=9))
    return fig

def load_data(source, **kw) -> pd.DataFrame:
    if source == "upload":
        f = kw["file"]
        if f.name.lower().endswith(".csv"):
            raw = f.read(4096).decode("utf-8", errors="ignore"); f.seek(0)
            sep = ";" if raw.count(";") > raw.count(",") else ","
            return pd.read_csv(f, sep=sep)
        return pd.read_excel(f)
    elif source == "url":
        return pd.read_csv(kw["url"])
    elif source == "s3":
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        # Bucket public (pas de credentials) → mode anonyme
        if not kw.get("key") and not kw.get("secret"):
            s3 = boto3.client("s3", region_name=kw["region"],
                              config=Config(signature_version=UNSIGNED))
        else:
            s3 = boto3.client("s3", aws_access_key_id=kw["key"],
                              aws_secret_access_key=kw["secret"],
                              region_name=kw["region"])
        obj = s3.get_object(Bucket=kw["bucket"], Key=kw["path"])
        raw = obj["Body"].read()
        if kw["path"].endswith(".csv"):    return pd.read_csv(io.BytesIO(raw))
        elif kw["path"].endswith(".parquet"): return pd.read_parquet(io.BytesIO(raw))
        else: return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    elif source == "azure":
        from azure.storage.blob import BlobServiceClient
        client = BlobServiceClient.from_connection_string(kw["conn"])
        raw    = client.get_blob_client(kw["container"],kw["blob"]).download_blob().readall()
        return pd.read_csv(io.BytesIO(raw)) if kw["blob"].endswith(".csv") else pd.read_parquet(io.BytesIO(raw))
    elif source == "gcs":
        from google.cloud import storage
        from google.oauth2 import service_account
        creds  = service_account.Credentials.from_service_account_info(json.loads(kw["creds"]))
        client = storage.Client(credentials=creds)
        raw    = client.bucket(kw["bucket"]).blob(kw["path"]).download_as_bytes()
        return pd.read_csv(io.BytesIO(raw)) if kw["path"].endswith(".csv") else pd.read_parquet(io.BytesIO(raw))
    elif source == "postgres":
        import sqlalchemy
        url    = f"postgresql://{kw['user']}:{kw['password']}@{kw['host']}:{kw['port']}/{kw['db']}"
        engine = sqlalchemy.create_engine(url)
        return pd.read_sql(kw["query"], engine)
    elif source == "mysql":
        import sqlalchemy
        url    = f"mysql+pymysql://{kw['user']}:{kw['password']}@{kw['host']}:{kw['port']}/{kw['db']}"
        engine = sqlalchemy.create_engine(url)
        return pd.read_sql(kw["query"], engine)


# ══════════════════════════════════════════════════════════════
# STATE INIT
# ══════════════════════════════════════════════════════════════

for k,v in [("step",1),("df",None),("result",None),("rules",[]),
            ("source_name","dataset"),("source_type","upload"),
            ("freshness_h",24),("alert_t",70),
            ("dbx_enabled",False),("dbx_workspace",""),("dbx_token",""),("dbx_cluster",""),
            ("admin_mode",False)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════
# ÉCRAN LOGIN
# ══════════════════════════════════════════════════════════════

if not auth.is_logged_in(st.session_state):

    # Détecter le mode admin (?admin dans l'URL ou bouton discret)
    params = st.query_params
    if params.get("admin") == "1":
        st.session_state.admin_mode = True

    if st.session_state.admin_mode:
        # ── ÉCRAN LOGIN ADMIN ──────────────────────────────────
        st.markdown("""
        <div class="login-wrap">
          <div class="login-logo">
            <div class="login-logo-icon" style="background:#DC2626;">🔐</div>
            <div class="login-title">Administration</div>
            <div class="login-sub">Accès réservé aux administrateurs</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        c1,c2,c3 = st.columns([1,1.4,1])
        with c2:
            st.markdown('<div class="login-card">', unsafe_allow_html=True)
            adm_user = st.text_input("Identifiant admin", placeholder="admin", key="adm_u")
            adm_pass = st.text_input("Mot de passe", type="password", key="adm_p")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Accéder à l'admin", type="primary", width='stretch'):
                user = auth.verify_login(adm_user, adm_pass)
                if user and user.get("role") == "admin":
                    st.session_state["user"] = user
                    st.rerun()
                elif user:
                    st.markdown('<div class="alert alert-err">⛔ Compte non administrateur.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert alert-err">❌ Identifiant ou mot de passe incorrect.</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            if st.button("← Retour au login client", key="back_login"):
                st.session_state.admin_mode = False
                st.query_params.clear()
                st.rerun()
        st.stop()

    else:
        # ── ÉCRAN LOGIN CLIENT ─────────────────────────────────
        st.markdown("""
        <div class="login-wrap">
          <div class="login-logo">
            <div class="login-logo-icon">⬡</div>
            <div class="login-title">DataQuality Agent</div>
            <div class="login-sub">Connectez-vous pour accéder à l'outil</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        c1,c2,c3 = st.columns([1,1.4,1])
        with c2:
            st.markdown('<div class="login-card">', unsafe_allow_html=True)
            username = st.text_input("Identifiant", placeholder="votre identifiant")
            password = st.text_input("Mot de passe", type="password", placeholder="••••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Se connecter", type="primary", width='stretch'):
                if auth.login(st.session_state, username, password):
                    st.rerun()
                else:
                    st.markdown('<div class="alert alert-err">Identifiant ou mot de passe incorrect.</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            # Lien admin discret en bas — invisible pour un client qui ne sait pas
            st.markdown("""
            <div style="text-align:center;margin-top:16px;">
              <span style="color:var(--dim);font-size:0.72rem;">Accès sur invitation · DataQuality Agent v3</span>
            </div>
            <div style="text-align:center;margin-top:32px;">
              <a href="?admin=1" style="color:var(--border2);font-size:0.6rem;
                 text-decoration:none;letter-spacing:1px;">· · ·</a>
            </div>
            """, unsafe_allow_html=True)
        st.stop()


# ══════════════════════════════════════════════════════════════
# PAGE ADMIN (si connecté en tant qu'admin via ?admin=1)
# ══════════════════════════════════════════════════════════════

user = auth.get_current_user(st.session_state)

if st.session_state.get("admin_mode") and user and user.get("role") == "admin":

    def _gen_pwd(n=12):
        chars = string.ascii_letters + string.digits + "!@#$"
        return ''.join(secrets_mod.choice(chars) for _ in range(n))

    def _fmt_date(s):
        if not s: return "—"
        try:
            from datetime import datetime as _dt
            return _dt.fromisoformat(s).strftime("%d/%m/%Y %H:%M")
        except: return s

    users_list = auth.list_users()
    clients_l  = [u for u in users_list if u["role"] == "client"]
    admins_l   = [u for u in users_list if u["role"] == "admin"]
    actifs_l   = [u for u in users_list if u.get("last_login")]

    # NAV admin
    st.markdown(f"""
    <div class="topnav" style="border-bottom:2px solid #FECACA;">
      <div class="brand">
        <div class="brand-icon" style="background:#DC2626;">🔐</div>
        <div>
          <div class="brand-name">DataQuality Agent</div>
          <div class="brand-tag">Administration</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:10px;">
        <span style="background:#FEF2F2;border:1px solid #FECACA;border-radius:20px;
          padding:4px 12px;font-size:0.65rem;font-weight:700;color:#DC2626;
          font-family:'JetBrains Mono',monospace;">ADMIN</span>
        <span class="user-pill">👤 {user['username']}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Stats
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:28px;">
      <div style="background:var(--surface);border:1.5px solid var(--border);border-radius:14px;padding:20px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
        <div style="font-family:'Instrument Serif',serif;font-size:2rem;">{len(users_list)}</div>
        <div style="font-size:0.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;font-weight:600;">Comptes total</div>
      </div>
      <div style="background:var(--surface);border:1.5px solid var(--border);border-radius:14px;padding:20px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
        <div style="font-family:'Instrument Serif',serif;font-size:2rem;color:var(--accent2);">{len(clients_l)}</div>
        <div style="font-size:0.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;font-weight:600;">Clients</div>
      </div>
      <div style="background:var(--surface);border:1.5px solid var(--border);border-radius:14px;padding:20px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
        <div style="font-family:'Instrument Serif',serif;font-size:2rem;color:var(--ok);">{len(actifs_l)}</div>
        <div style="font-size:0.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;font-weight:600;">Ont connecté</div>
      </div>
      <div style="background:var(--surface);border:1.5px solid var(--border);border-radius:14px;padding:20px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
        <div style="font-family:'Instrument Serif',serif;font-size:2rem;color:var(--danger);">{len(admins_l)}</div>
        <div style="font-size:0.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;font-weight:600;">Admins</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_main, col_side = st.columns([2.2, 1])

    with col_main:
        st.markdown('<div class="card-label">Comptes utilisateurs</div>', unsafe_allow_html=True)

        # Tableau des comptes
        rows = ""
        for u in sorted(users_list, key=lambda x: x.get("last_login") or "", reverse=True):
            role_color = "#DC2626" if u["role"]=="admin" else "var(--accent2)"
            role_bg    = "#FEF2F2" if u["role"]=="admin" else "var(--accent-l)"
            role_border= "#FECACA" if u["role"]=="admin" else "#C7D2FE"
            dot = '<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--ok);margin-right:5px;"></span>' if u.get("last_login") else ""
            status = f"{dot}Actif" if u.get("last_login") else '<span style="color:var(--dim);">Jamais connecté</span>'
            rows += f"""<tr>
              <td><strong>{u['username']}</strong></td>
              <td><span style="background:{role_bg};border:1px solid {role_border};color:{role_color};
                border-radius:20px;padding:2px 10px;font-size:0.62rem;font-weight:700;
                font-family:'JetBrains Mono',monospace;">{u['role'].upper()}</span></td>
              <td style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;">{_fmt_date(u.get('last_login'))}</td>
              <td>{status}</td>
            </tr>"""

        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;">
          <thead>
            <tr>
              <th style="background:var(--surface2);padding:10px 14px;text-align:left;font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid var(--border);">Utilisateur</th>
              <th style="background:var(--surface2);padding:10px 14px;text-align:left;font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid var(--border);">Rôle</th>
              <th style="background:var(--surface2);padding:10px 14px;text-align:left;font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid var(--border);">Dernière connexion</th>
              <th style="background:var(--surface2);padding:10px 14px;text-align:left;font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid var(--border);">Statut</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card-label">Actions</div>', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["🗑 Supprimer", "🔑 Réinitialiser mot de passe", "👑 Changer rôle"])

        with tab1:
            deletable = [u["username"] for u in users_list if u["username"] != user["username"]]
            if deletable:
                del_u = st.selectbox("Compte à supprimer", deletable, key="del_u")
                if st.button("🗑 Confirmer suppression", type="primary"):
                    if auth.delete_user(del_u):
                        st.markdown(f'<div class="alert alert-ok">✅ {del_u} supprimé.</div>', unsafe_allow_html=True)
                        st.rerun()
            else:
                st.markdown('<div class="alert alert-info">Aucun compte à supprimer.</div>', unsafe_allow_html=True)

        with tab2:
            reset_u   = st.selectbox("Compte", [u["username"] for u in users_list], key="rst_u")
            manual_p  = st.text_input("Nouveau mot de passe (vide = auto)", type="password", key="rst_p")
            if st.button("🔑 Réinitialiser", type="primary"):
                new_p = manual_p or _gen_pwd()
                if auth.change_password(reset_u, new_p):
                    st.markdown(f'<div class="alert alert-ok">✅ Mot de passe réinitialisé pour <strong>{reset_u}</strong></div>', unsafe_allow_html=True)
                    if not manual_p:
                        st.markdown('<div style="background:#1C1917;border-radius:10px;padding:14px;font-family:JetBrains Mono,monospace;font-size:1rem;color:#34D399;text-align:center;letter-spacing:2px;">' + str(new_p) + '</div>', unsafe_allow_html=True)

        with tab3:
            other_users = [u["username"] for u in users_list if u["username"] != user["username"]]
            if other_users:
                role_u = st.selectbox("Compte", other_users, key="role_u")
                cur_role = next((u["role"] for u in users_list if u["username"]==role_u), "client")
                new_role_val = st.selectbox("Nouveau rôle", ["client","admin"],
                               index=0 if cur_role=="client" else 1, key="role_v")
                if st.button("💾 Changer", type="primary"):
                    all_u = auth._load_users()
                    all_u[role_u]["role"] = new_role_val
                    auth._save_users(all_u)
                    st.markdown(f'<div class="alert alert-ok">✅ {role_u} → {new_role_val}</div>', unsafe_allow_html=True)
                    st.rerun()

    with col_side:
        st.markdown('<div class="card-label">Créer un compte</div>', unsafe_allow_html=True)
        with st.container():
            new_u    = st.text_input("Identifiant", placeholder="prenom.nom", key="new_u")
            new_r    = st.selectbox("Rôle", ["client","admin"], key="new_r")
            auto_p   = st.checkbox("Générer mot de passe auto", value=True, key="auto_p")
            manual_np = "" if auto_p else st.text_input("Mot de passe", type="password", key="mnl_p")
            if st.button("➕ Créer", type="primary", width='stretch'):
                if not new_u:
                    st.markdown('<div class="alert alert-err">Identifiant requis.</div>', unsafe_allow_html=True)
                elif auth.user_exists(new_u):
                    st.markdown(f'<div class="alert alert-err">{new_u} existe déjà.</div>', unsafe_allow_html=True)
                else:
                    pwd = _gen_pwd() if auto_p else manual_np
                    if pwd:
                        auth.create_user(new_u, pwd, role=new_r)
                        st.markdown(f'<div class="alert alert-ok">✅ Compte <strong>{new_u}</strong> créé !</div>', unsafe_allow_html=True)
                        if auto_p:
                            st.markdown('<div style="background:#1C1917;border-radius:10px;padding:14px 18px;font-family:JetBrains Mono,monospace;font-size:1.1rem;color:#34D399;letter-spacing:3px;text-align:center;margin:8px 0;">' + str(pwd) + '</div>', unsafe_allow_html=True)
                            st.markdown('<div style="text-align:center;font-size:0.72rem;color:var(--muted);">Transmettez ce mot de passe au client</div>', unsafe_allow_html=True)
                        st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card-label">Email invitation</div>', unsafe_allow_html=True)
        app_url_admin = st.text_input("URL app", value="https://ton-app.streamlit.app", key="app_url_a")
        inv_user = st.text_input("Identifiant client", key="inv_u")
        if st.button("📋 Générer email", width='stretch'):
            if inv_user:
                tpl = f"""Objet : Votre accès DataQuality Agent

Bonjour,

Votre accès est prêt.

🔗 {app_url_admin}
👤 Identifiant : {inv_user}
🔑 Mot de passe : [voir séparément]

Cordialement,
DataQuality Agent"""
                st.text_area("", value=tpl, height=200, key="email_tpl")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 Déconnexion admin", width='stretch'):
            auth.logout(st.session_state)
            st.session_state.admin_mode = False
            st.query_params.clear()
            st.rerun()

    st.stop()


# ══════════════════════════════════════════════════════════════
# UTILISATEUR CONNECTÉ — NAV
# ══════════════════════════════════════════════════════════════

step = st.session_state.step

labels = ["Source","Règles","Analyse","Rapport"]
steps_html = ""
for i, label in enumerate(labels, 1):
    cls  = "done" if i < step else ("active" if i == step else "")
    icon = "✓" if i < step else str(i)
    steps_html += f'<div class="step {cls}"><span class="step-num">{icon}</span>{label}</div>'
    if i < 4: steps_html += '<span class="step-sep">›</span>'

engine_label = "PySpark" if SPARK_AVAILABLE else "Pandas"

st.markdown(f"""
<div class="topnav">
  <div class="brand">
    <div class="brand-icon">⬡</div>
    <div>
      <div class="brand-name">DataQuality Agent</div>
      <div class="brand-tag">v3 · One-Shot · 0 donnée stockée</div>
    </div>
  </div>
  <div class="steps">{steps_html}</div>
  <div class="nav-right">
    <span class="engine-pill">⚡ {engine_label}</span>
    <span class="user-pill">👤 {user['username']}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Bouton logout dans un coin discret
with st.sidebar:
    if st.button("Déconnexion"):
        auth.logout(st.session_state)
        st.rerun()


# ══════════════════════════════════════════════════════════════
# ÉTAPE 1 — SOURCE
# ══════════════════════════════════════════════════════════════

if step == 1:
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">Audit · Scoring · Rapport</div>
      <div class="hero-title">Qualité de données<br><span>en 3 minutes</span></div>
      <div class="hero-sub">Connectez votre source, définissez vos règles,<br>obtenez un score sur 9 dimensions.</div>
      <div class="hero-stats">
        <div class="hero-stat"><div class="hero-stat-n">9</div><div class="hero-stat-l">Dimensions</div></div>
        <div class="hero-stat"><div class="hero-stat-n">0</div><div class="hero-stat-l">Donnée stockée</div></div>
        <div class="hero-stat"><div class="hero-stat-n">6</div><div class="hero-stat-l">Connecteurs</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Connexion Databricks (optionnel) ──────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('''
    <div class="dbx-card">
      <div class="dbx-header">
        <div class="dbx-icon">🧱</div>
        <div>
          <div class="dbx-title">Databricks — Engine PySpark (optionnel)</div>
          <div class="dbx-sub">Si vous avez un workspace Databricks, le scoring tournera sur votre cluster. Vos données ne quittent pas votre infra.</div>
        </div>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    dbx_on = st.toggle("Activer Databricks", value=st.session_state.dbx_enabled, key="dbx_toggle")
    st.session_state.dbx_enabled = dbx_on

    if dbx_on:
        dc1, dc2 = st.columns(2)
        st.session_state.dbx_workspace = dc1.text_input(
            "Workspace URL",
            value=st.session_state.dbx_workspace,
            placeholder="https://adb-xxxx.azuredatabricks.net",
            help="URL de votre workspace Databricks"
        )
        st.session_state.dbx_token = dc2.text_input(
            "Personal Access Token",
            value=st.session_state.dbx_token,
            type="password",
            help="Générer dans Databricks : Settings → Developer → Access tokens"
        )
        st.session_state.dbx_cluster = st.text_input(
            "Cluster ID",
            value=st.session_state.dbx_cluster,
            placeholder="0123-456789-abcdefgh",
            help="Dans Databricks : Compute → votre cluster → Advanced → Tags → ClusterId"
        )
        if st.button("🔌 Tester la connexion", type="primary"):
            if not st.session_state.dbx_workspace or not st.session_state.dbx_token:
                st.markdown('<div class="alert alert-err">⚠️ Workspace URL et Token requis.</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Test de connexion…"):
                    try:
                        import requests as req
                        url = st.session_state.dbx_workspace.rstrip("/") + "/api/2.0/clusters/get"
                        headers = {"Authorization": f"Bearer {st.session_state.dbx_token}"}
                        resp = req.get(url, headers=headers, params={"cluster_id": st.session_state.dbx_cluster}, timeout=8)
                        if resp.status_code == 200:
                            cluster_info = resp.json()
                            state = cluster_info.get("state","")
                            name  = cluster_info.get("cluster_name","cluster")
                            if state == "RUNNING":
                                st.markdown(f'<div class="alert alert-ok">✅ Connecté — <strong>{name}</strong> est actif. Le scoring utilisera PySpark.</div>', unsafe_allow_html=True)
                                st.session_state.dbx_connected = True
                            else:
                                st.markdown(f'<div class="alert alert-warn">⚠️ Cluster <strong>{name}</strong> en état <strong>{state}</strong>. Demarrez-le avant de lancer l analyse.</div>', unsafe_allow_html=True)
                                st.session_state.dbx_connected = False
                        elif resp.status_code == 401:
                            st.markdown('<div class="alert alert-err">❌ Token invalide ou expiré.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="alert alert-err">❌ Erreur {resp.status_code} — verifiez l URL et le Cluster ID.</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="alert alert-err">❌ Connexion impossible : {e}</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:10px;
                    padding:14px 16px;margin-top:10px;font-size:0.78rem;color:var(--muted);">
          <strong style="color:var(--text2);">Comment trouver votre Cluster ID ?</strong><br>
          Databricks → Compute → cliquer sur votre cluster → onglet <em>Configuration</em> → <em>Advanced options</em> → <em>Tags</em> → valeur de <code>ClusterId</code><br><br>
          <strong style="color:var(--text2);">Comment créer un Token ?</strong><br>
          Databricks → cliquer sur votre avatar → <em>Settings</em> → <em>Developer</em> → <em>Access tokens</em> → <em>Generate new token</em>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert alert-info" style="font-size:0.78rem;">ℹ️ Mode Pandas — fonctionne sans Databricks. Limite recommandée : 5M lignes.</div>', unsafe_allow_html=True)

    st.markdown('<div class="card-label">Source de données</div>', unsafe_allow_html=True)

    sources = [
        ("📁","CSV / Excel","Upload direct","upload"),
        ("🔗","URL / Drive","Lien public CSV","url"),
        ("🟠","Amazon S3","Bucket + credentials","s3"),
        ("🟦","Azure Blob","Connection string","azure"),
        ("🟡","Google Cloud","GCS + service account","gcs"),
        ("🐘","PostgreSQL / MySQL","Host + requête SQL","postgres"),
    ]

    sel = st.session_state.source_type
    cols6 = st.columns(3)
    for i,(icon,name,desc,key) in enumerate(sources):
        with cols6[i%3]:
            selected = "selected" if sel == key else ""
            st.markdown(f"""
            <div class="src-card {selected}">
              <div class="src-icon">{icon}</div>
              <div class="src-name">{name}</div>
              <div class="src-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)
            if st.button("Sélectionner", key=f"src_{key}", use_container_width=True):
                st.session_state.source_type = key; st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    src  = st.session_state.source_type
    df   = None

    if src == "upload":
        st.markdown('<div class="card-label">📁 Upload fichier</div>', unsafe_allow_html=True)
        f = st.file_uploader("Fichier", type=["csv","xlsx","xls"], label_visibility="collapsed")
        if f:
            with st.spinner("Lecture…"):
                try:
                    df = load_data("upload", file=f)
                    st.session_state.source_name = f.name
                except Exception as e:
                    st.error(f"Erreur : {e}")

    elif src == "url":
        st.markdown('<div class="card-label">🔗 URL publique</div>', unsafe_allow_html=True)
        url = st.text_input("URL du fichier CSV", placeholder="https://… ou https://drive.google.com/uc?id=…")
        if st.button("Charger", type="primary") and url:
            with st.spinner("Chargement…"):
                try:
                    df = load_data("url", url=url)
                    st.session_state.source_name = url.split("/")[-1] or "url_dataset"
                except Exception as e:
                    st.error(f"Erreur : {e}")

    elif src == "s3":
        st.markdown('<div class="card-label">🟠 Amazon S3</div>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        bucket=c1.text_input("Bucket"); path=c2.text_input("Chemin fichier")
        key=st.text_input("Access Key ID",type="password"); secret=st.text_input("Secret Access Key",type="password")
        region=st.text_input("Région",value="eu-west-1")
        if st.button("Connecter S3",type="primary"):
            with st.spinner("Connexion S3…"):
                try:
                    df=load_data("s3",bucket=bucket,path=path,key=key,secret=secret,region=region)
                    st.session_state.source_name=path.split("/")[-1]
                except Exception as e: st.error(f"Erreur S3 : {e}")

    elif src == "azure":
        st.markdown('<div class="card-label">🟦 Azure Blob</div>', unsafe_allow_html=True)
        conn=st.text_input("Connection String",type="password")
        c1,c2=st.columns(2); cont=c1.text_input("Container"); blob=c2.text_input("Blob")
        if st.button("Connecter Azure",type="primary"):
            with st.spinner("Connexion Azure…"):
                try:
                    df=load_data("azure",conn=conn,container=cont,blob=blob)
                    st.session_state.source_name=blob.split("/")[-1]
                except Exception as e: st.error(f"Erreur Azure : {e}")

    elif src == "gcs":
        st.markdown('<div class="card-label">🟡 Google Cloud Storage</div>', unsafe_allow_html=True)
        c1,c2=st.columns(2); bkt=c1.text_input("Bucket"); blb=c2.text_input("Chemin fichier")
        creds=st.text_area("Service Account JSON",height=80)
        if st.button("Connecter GCS",type="primary"):
            with st.spinner("Connexion GCS…"):
                try:
                    df=load_data("gcs",bucket=bkt,path=blb,creds=creds)
                    st.session_state.source_name=blb.split("/")[-1]
                except Exception as e: st.error(f"Erreur GCS : {e}")

    elif src == "postgres":
        st.markdown('<div class="card-label">🐘 PostgreSQL / MySQL</div>', unsafe_allow_html=True)
        db_type = st.selectbox("Base de données", ["PostgreSQL","MySQL"])
        c1,c2,c3=st.columns([3,1,1])
        host=c1.text_input("Host",placeholder="localhost")
        port=c2.text_input("Port",value="5432" if db_type=="PostgreSQL" else "3306")
        db=c3.text_input("Base")
        c4,c5=st.columns(2)
        user=c4.text_input("Utilisateur"); pwd=c5.text_input("Mot de passe",type="password")
        query=st.text_area("Requête SQL",value="SELECT * FROM ma_table LIMIT 50000",height=68)
        src_key = "postgres" if db_type == "PostgreSQL" else "mysql"
        if st.button(f"Connecter {db_type}",type="primary"):
            with st.spinner("Connexion…"):
                try:
                    df=load_data(src_key,host=host,port=port,db=db,user=user,password=pwd,query=query)
                    st.session_state.source_name=f"{db_type.lower()}_query"
                except Exception as e: st.error(f"Erreur {db_type} : {e}")

    # ── Données de démo ────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        if st.button("🎲  Données de démo", width='stretch'):
            np.random.seed(42); n=600
            df = pd.DataFrame({
                "id":       range(1,n+1),
                "email":    [f"user{i}@mail.com" if i%8!=0 else "invalid" for i in range(n)],
                "phone":    [f"+336{i:08d}" if i%10!=0 else "abc" for i in range(n)],
                "age":      [np.random.randint(18,80) if i%15!=0 else -5 for i in range(n)],
                "price":    np.random.exponential(100,n).round(2),
                "status":   np.random.choice(["active","ACTIVE","Active","inactive",None],n,p=[.3,.2,.1,.35,.05]),
                "country":  np.random.choice(["France","FRANCE","france","Germany","N/A"],n,p=[.3,.15,.1,.4,.05]),
                "created_at": pd.date_range("2024-01-01",periods=n,freq="1h").astype(str),
            })
            df = pd.concat([df,df.sample(25)],ignore_index=True)
            st.session_state.source_name = "demo_ecommerce.csv"

    if df is not None:
        st.session_state.df = df
        st.markdown(f"""
        <div class="alert alert-ok">
          ✅ <strong>{st.session_state.source_name}</strong> — {len(df):,} lignes × {len(df.columns)} colonnes
        </div>""", unsafe_allow_html=True)
        with st.expander("Aperçu", expanded=False):
            st.dataframe(df.head(8), width='stretch')
        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns([1,2,1])
        with c2:
            if st.button("Suivant — Règles métier →", type="primary", width='stretch'):
                st.session_state.step = 2; st.rerun()


# ══════════════════════════════════════════════════════════════
# ÉTAPE 2 — RÈGLES MÉTIER
# ══════════════════════════════════════════════════════════════

elif step == 2:
    df = st.session_state.df
    if df is None: st.session_state.step=1; st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Help box explicative
    st.markdown('''
    <div class="help-box">
      <div class="help-box-title">💡 Comment fonctionnent les règles métier ?</div>
      <div class="help-box-body">
        Une règle métier vérifie une contrainte sur <strong>chaque ligne</strong> de votre dataset.
        Choisissez une colonne, un opérateur, et une valeur seuil — l'engine compte le nombre de violations.<br><br>
        <strong>Exemples concrets :</strong>
        <div class="help-example">
          age &gt;= 0           → pas d'âge négatif<br>
          price &gt; 0          → prix toujours positif<br>
          status == "active"    → statut valide uniquement<br>
          discount &lt;= 100    → remise max 100%
        </div>
        Les règles s'ajoutent à la dimension <strong>Cohérence</strong> (15% du score global).
        Vous pouvez passer cette étape si vous n'avez pas de contraintes métier spécifiques.
      </div>
    </div>
    ''', unsafe_allow_html=True)

    cl, cr = st.columns([1.1, 1])

    with cl:
        st.markdown('<div class="sec-title">Règles métier</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-sub">Définissez vos contraintes sans écrire de code.</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Créer une règle</div>', unsafe_allow_html=True)

        all_cols = df.columns.tolist()
        rule_name = st.text_input("Nom de la règle", placeholder="ex : Prix toujours positif")
        c1,c2,c3 = st.columns([2,1.5,1.5])
        rule_col = c1.selectbox("Colonne", all_cols)
        rule_op  = c2.selectbox("Opérateur", [">",">=","<","<=","==","!="])
        rule_val = c3.text_input("Valeur", placeholder="0")
        rule_sev = st.select_slider("Sévérité", options=["low","medium","high"], value="medium")

        if st.button("➕ Ajouter", type="primary", width='stretch'):
            if rule_name and rule_val:
                try:    float(rule_val); cond = f"`{rule_col}` {rule_op} {rule_val}"
                except: cond = f"`{rule_col}` {rule_op} '{rule_val}'"
                st.session_state.rules.append({
                    "name":      rule_name,
                    "condition": cond,
                    "column":    rule_col,
                    "severity":  rule_sev,
                    "operator":  rule_op,
                    "value":     rule_val,
                }); st.rerun()

        # Prédéfinies
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card-label">Règles suggérées</div>', unsafe_allow_html=True)
        numeric_cols = df.select_dtypes(include=["float64","int64"]).columns.tolist()
        presets = []
        for col in numeric_cols:
            if any(kw in col.lower() for kw in ["price","prix","amount","montant","cost"]):
                presets.append({"name":f"{col} positif","condition":f"`{col}` >= 0","column":col,"severity":"high"})
            if "age" in col.lower():
                presets.append({"name":f"{col} valide (0-120)","condition":f"`{col}` >= 0 and `{col}` <= 120","column":col,"severity":"high"})
        if presets:
            for p in presets:
                c1,c2 = st.columns([5,1])
                c1.markdown(f"""
                <div class="rule-item">
                  <div class="rule-name-t">{p['name']}</div>
                  <div class="rule-cond-t">{p['condition']}</div>
                </div>""", unsafe_allow_html=True)
                if c2.button("＋",key=f"p_{p['name']}"):
                    if p not in st.session_state.rules:
                        st.session_state.rules.append(p); st.rerun()
        else:
            st.markdown('<div style="color:var(--muted);font-size:0.8rem;">Aucune colonne numérique détectée.</div>', unsafe_allow_html=True)

    with cr:
        st.markdown(f'<div class="sec-title">Règles actives ({len(st.session_state.rules)})</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-sub">Les règles seront vérifiées sur chaque ligne.</div>', unsafe_allow_html=True)
        if not st.session_state.rules:
            st.markdown("""
            <div style="background:var(--surface);border:1px dashed var(--border2);border-radius:12px;
                        padding:44px 20px;text-align:center;color:var(--dim);">
              <div style="font-size:1.8rem;margin-bottom:6px;">📋</div>
              <div style="font-size:0.82rem;">Aucune règle — vous pouvez passer cette étape</div>
            </div>""", unsafe_allow_html=True)
        else:
            for i, rule in enumerate(st.session_state.rules):
                sc_col = {"high":"#EF4444","medium":"#F59E0B","low":"#10B981"}.get(rule["severity"],"#64748B")
                c1,c2 = st.columns([5,1])
                c1.markdown(f"""
                <div class="rule-item">
                  <div class="rule-name-t">{rule['name']}</div>
                  <div class="rule-cond-t">{rule['condition']}</div>
                  <div style="font-size:.65rem;color:{sc_col};margin-top:3px;">● {rule['severity'].upper()}</div>
                </div>""", unsafe_allow_html=True)
                if c2.button("🗑",key=f"del_{i}"):
                    st.session_state.rules.pop(i); st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card-label">Paramètres</div>', unsafe_allow_html=True)
        st.session_state.freshness_h = st.slider("Seuil fraîcheur (heures)", 1, 720, st.session_state.freshness_h)
        st.session_state.alert_t     = st.slider("Seuil d'alerte (score)", 0, 100, st.session_state.alert_t)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        ca,cb = st.columns(2)
        if ca.button("← Retour", width='stretch'):
            st.session_state.step=1; st.rerun()
        if cb.button("Lancer l'analyse →", type="primary", width='stretch'):
            st.session_state.result=None; st.session_state.step=3; st.rerun()


# ══════════════════════════════════════════════════════════════
# ÉTAPE 3 — DASHBOARD
# ══════════════════════════════════════════════════════════════

elif step == 3:
    df = st.session_state.df
    if df is None: st.session_state.step=1; st.rerun()

    if st.session_state.result is None:
        with st.spinner("Analyse en cours — 9 dimensions…"):
            # Connexion Databricks si activée
            spark_session = None
            if st.session_state.get("dbx_enabled") and st.session_state.get("dbx_connected"):
                try:
                    from databricks.connect import DatabricksSession
                    spark_session = DatabricksSession.builder \
                        .remote(
                            host=st.session_state.dbx_workspace,
                            token=st.session_state.dbx_token,
                            cluster_id=st.session_state.dbx_cluster,
                        ).getOrCreate()
                    df_spark = spark_session.createDataFrame(df)
                    st.markdown('<div class="alert alert-ok" style="font-size:.75rem;">⚡ Engine PySpark actif sur votre cluster Databricks</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="alert alert-warn" style="font-size:.75rem;">⚠️ Databricks Connect indisponible ({e}) — bascule sur Pandas</div>', unsafe_allow_html=True)
                    df_spark = None

            result = run_scoring(
                df=df_spark if spark_session and df_spark is not None else df,
                table_name=st.session_state.source_name,
                custom_rules=st.session_state.rules,
                freshness_threshold_hours=st.session_state.freshness_h,
                spark=spark_session,
            )
            st.session_state.result = result

    result = st.session_state.result
    g = result.global_score; gc = sc_hex(g)

    # Alerte
    at = st.session_state.alert_t
    if g >= 80:    cls,msg="alert-ok", f"✅ Score excellent ({g}/100) — données en très bonne santé"
    elif g >= at:  cls,msg="alert-warn",f"⚠️ Score acceptable ({g}/100) — points d'attention"
    else:          cls,msg="alert-err", f"🔴 Score insuffisant ({g}/100) — action requise"
    st.markdown(f'<div class="alert {cls}">{msg}</div>', unsafe_allow_html=True)

    # Engine badge
    eng_txt = f"⚡ Engine : {result.engine.upper()}"
    eng_cls = "alert-ok" if result.engine=="pyspark" else "alert-info"
    st.markdown(f'<div class="alert {eng_cls}" style="padding:8px 14px;font-size:.75rem;">{eng_txt}</div>', unsafe_allow_html=True)

    # Score + radar
    cs, cr = st.columns([1, 1.4])
    with cs:
        st.markdown(f"""
        <div class="score-ring">
          <div class="score-number" style="color:{gc};">{g}<span class="score-denom">/100</span></div>
          <div class="score-badge" style="color:{gc};border-color:{gc}33;background:{gc}11;">
            {se(g)} &nbsp; {sl(g)}
          </div>
          <div class="score-meta">{result.row_count:,} lignes · {result.col_count} colonnes<br>{result.table_name}</div>
        </div>""", unsafe_allow_html=True)
        high_c=len([i for i in result.issues if i.get("severity")=="high"])
        med_c =len([i for i in result.issues if i.get("severity")=="medium"])
        st.markdown(f"""
        <div class="metrics-row">
          <div class="metric"><div class="metric-val" style="color:#EF4444;">{high_c}</div><div class="metric-lbl">Critiques</div></div>
          <div class="metric"><div class="metric-val" style="color:#F59E0B;">{med_c}</div><div class="metric-lbl">Moyens</div></div>
          <div class="metric"><div class="metric-val">{len(result.issues)}</div><div class="metric-lbl">Total</div></div>
        </div>""", unsafe_allow_html=True)
    with cr:
        st.plotly_chart(radar_chart(result), width='stretch')

    # 9 dimensions
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card-label">9 dimensions</div>', unsafe_allow_html=True)
    dims_info=[
        ("Complétude",result.completeness,"20%","💧"),("Cohérence",result.consistency,"15%","✅"),
        ("Validité",result.validity,"15%","🔍"),("Unicité",result.uniqueness,"12%","🔑"),
        ("Fraîcheur",result.freshness,"10%","⏱"),("Distribution",result.distribution,"8%","📊"),
        ("Corrélation",result.correlation,"8%","🔗"),("Volumétrie",result.volumetry,"7%","📦"),
        ("Standard.",result.standardization,"5%","🧹"),
    ]
    cols9 = st.columns(9)
    for col,(name,score,weight,icon) in zip(cols9,dims_info):
        c=sc_cls(score); ch=sc_hex(score)
        with col:
            st.markdown(f"""
            <div class="dim-tile {c}">
              <div style="font-size:.9rem;">{icon}</div>
              <div class="dim-score" style="color:{ch};">{score}</div>
              <div class="dim-name">{name}</div>
              <div class="dim-weight">{weight}</div>
              <div class="dim-bar"><div class="dim-bar-fill" style="width:{score}%;background:{ch};"></div></div>
            </div>""", unsafe_allow_html=True)

    # Issues + colonnes
    st.markdown("<br>", unsafe_allow_html=True)
    high=[i for i in result.issues if i.get("severity")=="high"]
    med =[i for i in result.issues if i.get("severity")=="medium"]
    low =[i for i in result.issues if i.get("severity")=="low"]

    t1,t2,t3,t4 = st.tabs([
        f"🔴 Critiques ({len(high)})",f"🟡 Moyens ({len(med)})",
        f"🟢 Faibles ({len(low)})",f"📊 Par colonne",
    ])
    with t1: render_issues(high)
    with t2: render_issues(med)
    with t3: render_issues(low)
    with t4:
        cl,cr = st.columns(2)
        with cl:
            st.markdown('<div class="card-label">Score par colonne</div>', unsafe_allow_html=True)
            st.plotly_chart(bar_chart(result), width='stretch')
        with cr:
            st.markdown('<div class="card-label">Tableau</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([{
                "Colonne":c.name,"Complétude":f"{c.completeness}%",
                "Unicité":f"{c.uniqueness}%","Score":c.overall,"":se(c.overall),
            } for c in sorted(result.columns,key=lambda x:x.overall)]),
            width='stretch', hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        ca,cb = st.columns(2)
        if ca.button("← Modifier les règles",width='stretch'):
            st.session_state.result=None; st.session_state.step=2; st.rerun()
        if cb.button("Générer le rapport →",type="primary",width='stretch'):
            st.session_state.step=4; st.rerun()


# ══════════════════════════════════════════════════════════════
# ÉTAPE 4 — RAPPORT PDF
# ══════════════════════════════════════════════════════════════

elif step == 4:
    result = st.session_state.result
    if result is None: st.session_state.step=1; st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,2.5,1])
    with c2:
        g=result.global_score; gc=sc_hex(g)
        st.markdown(f"""
        <div style="text-align:center;padding:28px 0 20px;">
          <div style="font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;
                      color:var(--text);letter-spacing:-1px;">Rapport prêt</div>
          <div style="color:var(--muted);font-size:0.85rem;margin-top:4px;">
            Généré en mémoire · Aucune donnée transmise · Engine {result.engine.upper()}
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:16px;
                    padding:24px;margin-bottom:20px;text-align:center;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:3rem;
                      font-weight:500;color:{gc};letter-spacing:-2px;">{g}</div>
          <div style="color:var(--muted);font-size:.85rem;">Score global / 100</div>
          <div style="display:flex;justify-content:center;gap:28px;margin-top:18px;
                      padding-top:18px;border-top:1px solid var(--border);">
            <div style="text-align:center;">
              <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;">{result.row_count:,}</div>
              <div style="font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;">Lignes</div>
            </div>
            <div style="text-align:center;">
              <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;">{result.col_count}</div>
              <div style="font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;">Colonnes</div>
            </div>
            <div style="text-align:center;">
              <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:#EF4444;">{len([i for i in result.issues if i.get('severity')=='high'])}</div>
              <div style="font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;">Critiques</div>
            </div>
            <div style="text-align:center;">
              <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;">{len(result.custom_rules)}</div>
              <div style="font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;">Règles</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        with st.spinner("Génération du PDF…"):
            pdf_bytes = generate_pdf(result)

        if pdf_bytes:
            b64   = base64.b64encode(pdf_bytes).decode()
            fname = f"rapport_{result.table_name.replace('.','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.markdown(f"""
            <div style="text-align:center;margin-bottom:20px;">
              <a href="data:application/pdf;base64,{b64}" download="{fname}"
                 style="display:inline-flex;align-items:center;gap:10px;background:var(--accent);
                        color:white;padding:14px 28px;border-radius:12px;text-decoration:none;
                        font-family:'Syne',sans-serif;font-weight:700;font-size:.95rem;">
                ⬇️ &nbsp; Télécharger le rapport PDF
              </a>
            </div>
            <iframe src="data:application/pdf;base64,{b64}" width="100%" height="780px"
                    style="border:1px solid var(--border);border-radius:12px;"></iframe>
            """, unsafe_allow_html=True)
        else:
            st.warning("fpdf2 non installé — ajoutez-le à requirements.txt")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄  Nouvel audit", width='stretch'):
            for k in ["step","df","result","rules","source_name","source_type"]:
                if k in st.session_state: del st.session_state[k]
            st.rerun()
