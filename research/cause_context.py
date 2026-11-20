#!/usr/bin/env python3
"""Independent indicators of out-of-hospital and sudden death, Taiwan 2019-2025.

The cause-of-death analysis finds a large rise in vaguely certified deaths
(cardiogenic shock, cardiorespiratory failure, unspecified heart disease).
If these were real sudden deaths outside hospital, other statistics should
have moved as well.  This script assembles three national series that do not
depend on how the cause of death is certified or coded:

    1. patients in cardiac arrest before arrival, non-traumatic, transported by
       the fire services' ambulances (Ministry of the Interior statistics);
    2. deaths by place of death (household registration, by sex);
    3. judicial inquests by district prosecutors' offices (annual reports of the
       Institute of Forensic Medicine, table 1-12).

None of the three is published by age, so they describe all ages together.

Sources and checksums: data/SOURCES_taiwan_official.md.
Run from the repository root (reads downloaded files, hence -I):
    python3 -I research/cause_context.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

SRC = Path("data/context/2026-11-17/downloads")
OUT = Path("results_cause")
EMS = SRC / "moi_statis_ems_service_10401_11412" / "ems_service_monthly.csv"
OHCA_NON_TRAUMA, OHCA_TRAUMA = 13, 25           # the two columns headed 到院前心肺功能停止

# 全國地檢署相驗案件數, 法醫研究所「法醫鑑驗業務統計年報」表 1-12
# (2019-2020 from the 112 edition, 2021-2025 from the 114 edition; both read from the PDFs)
INQUESTS = {2019: 19008, 2020: 18906, 2021: 19186, 2022: 19562, 2023: 19545, 2024: 20494, 2025: 20207}


def ems_series() -> pd.DataFrame:
    with open(EMS, encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.reader(fh) if r]
    header = rows[0]
    if header[OHCA_NON_TRAUMA] != "到院前心肺功能停止" or header[OHCA_TRAUMA] != "到院前心肺功能停止":
        raise ValueError("unexpected column layout in the EMS file")
    monthly, annual = {}, {}
    for r in rows[1:]:
        m = re.fullmatch(r"(\d+)年(?: (\d+)月)?/ 區域別總計", r[0])
        if not m:
            continue
        year = int(m.group(1)) + 1911
        values = (int(r[OHCA_NON_TRAUMA]), int(r[OHCA_TRAUMA]))
        if m.group(2):
            a, b = monthly.get(year, (0, 0))
            monthly[year] = (a + values[0], b + values[1])
        else:
            annual[year] = values
    for year, v in monthly.items():                      # the annual rows must equal the sum of the twelve months
        if annual[year] != v:
            raise ValueError(f"EMS {year}: annual row {annual[year]} differs from the monthly sum {v}")
    return pd.DataFrame([(y, v[0], v[1]) for y, v in sorted(monthly.items())], columns=["year", "ohca_non_traumatic", "ohca_traumatic"])


def place_series() -> pd.DataFrame:
    out = []
    for roc in range(108, 115):
        path = SRC / f"moi_place_reg_{roc}" / f"moi_place_reg_{roc}.csv"
        d = pd.read_csv(path, encoding="utf-8-sig")
        if set(d["統計年度"]) != {roc}:
            raise ValueError(f"{path.name}: unexpected year")
        for sex, label in (("男", "Male"), ("女", "Female"), (None, "Total")):
            g = d if sex is None else d[d["性別"] == sex]
            by = g.groupby("地點")["人口數"].sum()
            total = int(by.sum())
            out.append((roc + 1911, label, total, int(by.get("醫院", 0)), int(by.get("住居所", 0)), int(by.get("長期照護或安養機構", 0)),
                        int(by.get("其他", 0)), int(by.get("國外", 0)), int(by.get("診所", 0))))
    p = pd.DataFrame(out, columns=["year", "sex", "deaths_registered", "hospital", "residence", "long_term_care", "other", "abroad", "clinic"])
    p["share_residence"] = p["residence"] / p["deaths_registered"]
    p["share_outside_hospital"] = 1 - p["hospital"] / p["deaths_registered"]
    return p


def main() -> None:
    ems = ems_series()
    place = place_series()
    ems.to_csv(OUT / "context_ems_ohca.csv", index=False)
    place.to_csv(OUT / "context_place_of_death.csv", index=False)
    deaths = place[place.sex == "Total"].set_index("year")["deaths_registered"]
    inq = pd.DataFrame([(y, n, int(deaths[y]), n / deaths[y]) for y, n in INQUESTS.items()],
                       columns=["year", "judicial_inquests", "deaths_registered", "inquests_per_death"])
    inq.to_csv(OUT / "context_inquests.csv", index=False)
    pd.set_option("display.width", 200)
    print(ems[ems.year >= 2015].to_string(index=False))
    print(place[place.sex != "Female"].round(4).to_string(index=False))
    print(inq.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
