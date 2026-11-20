#!/usr/bin/env python3
"""National deaths by year, sex, age group and cause group, Taiwan 2008-2025.

Source: Ministry of Health and Welfare, 死因統計（國際疾病分類標準之詳細分類）,
sheet "2008以後": one row per year x sex x ICD-10 code (3 or 4 characters, no
dot) with deaths by age group up to 100+.

Each ICD-10 code is assigned to exactly one analysis group (`assign_group`).
To check the reading of the ICD ranges, the script also rebuilds the
Ministry's own condensed categories from the detailed codes and compares
them with the Ministry's condensed file (42-code list, township level).

Note on comparability: from 2019 (ROC 108) Taiwan selects the underlying
cause by the 2016 edition of the ICD-10 rules; 2008-2018 follow the earlier
edition.  Cause-specific series therefore have a break between 2018 and 2019
and comparisons should stay within 2008-2018 or within 2019 onwards.

Run from the repository root (reads downloaded files, hence -I):
    python3 -I research/cause_prepare.py
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import openpyxl
import pandas as pd

SRC = Path("data/mohw/2026-11-17/downloads")
DETAIL_ZIP = SRC / "icd_detail" / "icd_detail.zip"
CONDENSED_ZIP = SRC / "condensed_42" / "all_cause_42.zip"
OUT = Path("results_cause")

AGE_COLUMNS = {  # workbook column -> lower bound of the age group
    "age_0_6days": 0, "age_7_27days": 0, "age_28_364days": 0, "age_1_4years": 1,
    **{f"age_{a}_{a + 4}years": a for a in range(5, 100, 5)}, "age_100+years": 100,
}
ALCOHOL = {"K860", "I426", "G312", "G621", "G721", "K292"}      # besides F10, K70, X45

GROUP_LABELS_ZH = {
    "covid19": "COVID-19", "cancer_oral_esophagus": "口腔癌及食道癌", "cancer_liver": "肝癌", "cancer_lung": "肺癌",
    "cancer_colorectal": "結腸直腸癌", "cancer_breast": "乳癌", "cancer_other": "其他惡性腫瘤",
    "heart_ischemic": "缺血性心臟病", "heart_other": "其他心臟疾病", "cerebrovascular": "腦血管疾病",
    "hypertensive": "高血壓性疾病", "diabetes": "糖尿病", "pneumonia_influenza": "肺炎及流感",
    "aspiration": "吸入性肺炎等", "chronic_lower_respiratory": "慢性下呼吸道疾病", "alcohol_specific": "酒精所致疾病",
    "liver_nonalcohol": "慢性肝病及病毒性肝炎（非酒精）", "kidney": "腎炎、腎病症候群及腎病變", "sepsis": "敗血症",
    "dementia_parkinson": "失智症、帕金森病及阿茲海默病", "senility_illdefined": "衰老及診斷欠明", "accidents": "事故傷害",
    "suicide": "蓄意自我傷害", "other": "其他",
}


def parse(icd: str) -> tuple[str, int, str]:
    """Letter, two-digit number and the full code; 'ZZZ' (pending inquest) -> ('Z', 99)."""
    icd = icd.strip().upper()
    if not re.fullmatch(r"[A-Z]\d{2}[0-9A-Z]?", icd):
        return "Z", 99, icd
    return icd[0], int(icd[1:3]), icd


def assign_group(icd: str) -> str:
    L, n, code = parse(icd)
    if L == "U" and n in (7, 9, 10):
        return "covid19"
    if (L == "F" and n == 10) or (L == "K" and n == 70) or (L == "X" and n == 45) or code in ALCOHOL:
        return "alcohol_specific"
    if L == "C" and n <= 97:
        if n <= 6 or n in (9, 10, 12, 13, 14, 15):
            return "cancer_oral_esophagus"
        if 18 <= n <= 21:
            return "cancer_colorectal"
        if n == 22:
            return "cancer_liver"
        if n in (33, 34):
            return "cancer_lung"
        if n == 50:
            return "cancer_breast"
        return "cancer_other"
    if L == "E" and 10 <= n <= 14:
        return "diabetes"
    if (L == "F" and 1 <= n <= 3) or (L == "G" and n in (20, 21, 30)):
        return "dementia_parkinson"
    if L == "I":
        if 10 <= n <= 15:
            return "hypertensive"
        if 20 <= n <= 25:
            return "heart_ischemic"
        if n == 1 or code == "I020" or 5 <= n <= 9 or n == 27 or 30 <= n <= 52:
            return "heart_other"
        if 60 <= n <= 69:
            return "cerebrovascular"
        return "other"
    if L == "J":
        if 10 <= n <= 18:
            return "pneumonia_influenza"
        if 40 <= n <= 47:
            return "chronic_lower_respiratory"
        if 66 <= n <= 69:
            return "aspiration"
        return "other"
    if (L == "K" and n in (73, 74)) or (L == "B" and 15 <= n <= 19):
        return "liver_nonalcohol"
    if L == "N" and (n <= 7 or 17 <= n <= 19 or 25 <= n <= 27):
        return "kidney"
    if L == "A" and n in (40, 41):
        return "sepsis"
    if L == "R":
        return "senility_illdefined"
    if L in ("V", "W") or (L == "X" and n <= 59) or (L == "Y" and n in (85, 86)):
        return "accidents"
    if (L == "X" and 60 <= n <= 84) or code == "Y870":
        return "suicide"
    return "other"


def condensed_category(icd: str) -> str | None:
    """The Ministry's own condensed categories (subset used for checking)."""
    L, n, code = parse(icd)
    if L == "C" and n <= 97:
        return "06"
    if L == "E" and 10 <= n <= 14:
        return "09"
    if L == "I" and 10 <= n <= 15:
        return "15"
    if L == "I" and (n == 1 or code == "I020" or 5 <= n <= 9 or 20 <= n <= 25 or n == 27 or 30 <= n <= 52):
        return "16"
    if L == "I" and 60 <= n <= 69:
        return "17"
    if L == "J" and 12 <= n <= 18:
        return "21"
    if L == "J" and 40 <= n <= 47:
        return "23"
    if L == "K" and n in (70, 73, 74):
        return "28"
    if L == "N" and (n <= 7 or 17 <= n <= 19 or 25 <= n <= 27):
        return "32"
    if L == "A" and n in (40, 41):
        return "03"
    if code == "U071":
        return "covid"
    if L in ("V", "W") or (L == "X" and n <= 59) or (L == "Y" and n in (85, 86)):
        return "accidents"
    if (L == "X" and 60 <= n <= 84) or code == "Y870":
        return "suicide"
    return None


def read_detail() -> pd.DataFrame:
    with zipfile.ZipFile(DETAIL_ZIP) as zf:
        member = next(i for i in zf.infolist() if i.filename.lower().endswith(".xlsx"))
        wb = openpyxl.load_workbook(io.BytesIO(zf.read(member)), read_only=True, data_only=True)
    ws = wb["2008以後"]
    rows = ws.iter_rows(values_only=True)
    header = [str(h) for h in next(rows)]
    expected = ["year", "sex", "ICD", "sum_age_groups"] + list(AGE_COLUMNS)
    if header != expected:
        raise ValueError(f"unexpected header: {header}")
    df = pd.DataFrame(list(rows), columns=header)
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)
    df["sex"] = df["sex"].astype(int).map({1: "Male", 2: "Female"})
    df["ICD"] = df["ICD"].astype(str).str.strip().str.upper()
    for col in ["sum_age_groups"] + list(AGE_COLUMNS):
        df[col] = pd.to_numeric(df[col]).fillna(0).astype(int)
    if not (df[list(AGE_COLUMNS)].sum(axis=1) == df["sum_age_groups"]).all():
        raise ValueError("age columns do not sum to the row total")
    return df


def read_condensed() -> pd.DataFrame:
    """National totals by year, sex and condensed code from the 42-code file."""
    frames = []
    with zipfile.ZipFile(CONDENSED_ZIP) as zf:
        for name in zf.namelist():
            m = re.fullmatch(r"(?:.*/)?dead(\d{2,3})\.txt", name)     # the .txt twins keep leading zeros in every year
            if not m or int(m.group(1)) < 97:
                continue
            d = pd.read_csv(io.BytesIO(zf.read(name)), dtype=str, encoding="utf-8-sig")
            d["N"] = d["N"].astype(int)
            roc = int(m.group(1))
            d["cause"] = d["cause"].str.zfill(2)
            shift = roc >= 109                                        # COVID-19 inserted as code 38
            d["cat"] = d["cause"].map(lambda c: {"38": "covid" if shift else "accidents", "39": "accidents" if shift else "suicide",
                                                  "40": "suicide" if shift else None}.get(c, c))
            g = d.groupby(["sex", "cat"], as_index=False)["N"].sum()
            g["year"] = roc + 1911
            frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    out["sex"] = out["sex"].map({"1": "Male", "2": "Female"})
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    detail = read_detail()
    detail["cause_group"] = detail["ICD"].map(assign_group)
    detail["condensed"] = detail["ICD"].map(condensed_category)

    long = detail.melt(id_vars=["year", "sex", "ICD", "cause_group"], value_vars=list(AGE_COLUMNS), var_name="age_col", value_name="deaths")
    long["age_lower"] = long["age_col"].map(AGE_COLUMNS)
    by_cause = long.groupby(["year", "sex", "age_lower", "cause_group"], as_index=False)["deaths"].sum()
    by_cause.to_csv(OUT / "deaths_by_cause.csv", index=False)

    icd_map = (detail.groupby(["ICD", "cause_group"], as_index=False)["sum_age_groups"].sum()
               .rename(columns={"sum_age_groups": "deaths_2008_2025"}).sort_values("ICD"))
    icd_map.to_csv(OUT / "icd_group_map.csv", index=False)

    totals = detail.groupby("year")["sum_age_groups"].sum()
    own = (detail.dropna(subset=["condensed"]).groupby(["year", "sex", "condensed"], as_index=False)["sum_age_groups"].sum()
           .rename(columns={"condensed": "cat", "sum_age_groups": "from_detail"}))
    official = read_condensed().rename(columns={"N": "from_condensed_file"})
    check = own.merge(official, on=["year", "sex", "cat"], how="left")
    check["difference"] = check["from_detail"] - check["from_condensed_file"]
    check.to_csv(OUT / "condensed_category_check.csv", index=False)

    meta = {
        "script": "research/cause_prepare.py",
        "inputs_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (DETAIL_ZIP, CONDENSED_ZIP)},
        "years": [int(detail["year"].min()), int(detail["year"].max())],
        "deaths_by_year": {int(k): int(v) for k, v in totals.items()},
        "distinct_icd_codes": int(detail["ICD"].nunique()),
        "cause_groups": GROUP_LABELS_ZH,
        "coding_break": "2008-2018: earlier ICD-10 selection rules; 2019 onwards: 2016 edition. Compare within each period only.",
        "condensed_check_max_abs_difference": int(check["difference"].abs().max()),
    }
    (OUT / "prepare_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    pd.set_option("display.width", 220)
    print("deaths by year:", meta["deaths_by_year"])
    print("distinct ICD codes:", meta["distinct_icd_codes"])
    print("check against the Ministry's condensed categories: max |difference| by category")
    print(check.groupby("cat")["difference"].agg(lambda x: int(x.abs().max())).to_string())
    share = by_cause[by_cause.year >= 2019].groupby("cause_group")["deaths"].sum().sort_values(ascending=False)
    print("\ndeaths 2019-2025 by analysis group:")
    print((share / share.sum() * 100).round(2).to_string())


if __name__ == "__main__":
    main()
