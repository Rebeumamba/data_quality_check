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
                        "column":col,"message":f"{neg:,} valeurs négatives"})
        for rule in self.custom_rules:
            try:
                n = (~df.eval(rule["condition"])).sum(); c += len(df); v += n
                if n > 0:
                    r.issues.append({"dimension":"consistency",
                        "severity":rule.get("severity","medium"),
                        "column":rule.get("column","custom"),
                        "message":f"Règle '{rule['name']}': {n:,} violations"})
            except: pass
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


print("✅ engine.py chargé")
print(f"   PySpark disponible : {SPARK_AVAILABLE}")
print(f"   Usage : result = run_scoring(df, table_name='mon_dataset', custom_rules=[...])")
