# 重現套件：〈疫情後臺灣成年死亡率回到趨勢了嗎？年齡別與死因別分析，兼與日本、韓國、香港比較〉

本套件收錄該文全部分析所用的程式、輸入資料與結果，以及該文的線上附錄。執行一個指令，就能由輸入資料重新算出該文分析所得的每一個數字、每一張表與每一張圖，並逐檔與稿件所載的結果比對。該文引自官方公布統計或其他研究的數字，不在重算之列。稿件正文不在套件內。

該文尚未刊登；刊登之前，套件的內容可能隨稿件修訂而更新。

*This package recomputes every number, table and figure that the article above (written in Chinese) derives from its own analysis, starting from the public data it is based on. Run `python3 reproduce.py` in this folder; the last step compares the recomputed results with the ones reported in the article. The article has not been published yet, and the package may change with it. An English summary is at the end of this file.*

## 內容

| 路徑 | 內容 |
|---|---|
| `reproduce.py` | 依序執行全部分析，最後以 `verify.py` 比對結果 |
| `verify.py` | 比對重算的結果與 `published/`，並核對套件內的檔案沒有被更動 |
| `research/` | 分析程式，共 13 支（見「程式與結果」） |
| `data/` | 輸入資料：人類死亡率資料庫（HMD）22 個地區的死亡數、暴露人年與時期生命表，以及內政部與衛生福利部的原始統計檔 |
| `published/` | 稿件所載的結果：`results_trend/` 與 `results_cause/` 是各步驟的結果，`generated/` 是文中的數字與表格，`figures/` 是四張圖 |
| `online_appendix/` | 該文的線上附錄（PDF 與 Word），含 13 張附表 |
| `DATA_SOURCES.md` | 每個資料檔的來源、取得日期與授權 |
| `LICENSE` | 程式碼的授權條款（MIT） |
| `CHECKSUMS.sha256` | 套件內其他各檔案的 SHA-256（清單不含這個檔案本身） |

## 重現方式

需要 Python 3.12 與 `requirements.txt` 所列的套件。圖中的文字使用 Liberation Sans 與 cwTeX Q Hei 兩種字型；沒有這兩種字型時圖仍會產生，只是外觀不同，數字不受影響。

```
python3 -m pip install -r requirements.txt
python3 reproduce.py
```

`reproduce.py` 是純 Python 程式，不需要 shell，在 Linux、macOS 與 Windows 的用法相同；系統若沒有 `python3` 這個指令，改用 `python`。可以從任何目錄執行；不在套件目錄時，指令要寫出 `reproduce.py` 的路徑，例如 `python3 路徑/reproduce.py`。

全部跑完約需 11 分鐘（24 核心的 Linux 主機，只安裝 `requirements.txt` 所列套件的全新 Python 環境，記憶體用量約 0.4 GB，2026-11-20 實測）。程式把結果寫到 `results_trend/`、`results_cause/` 與 `manuscript/`，不會更動 `published/`。最後一步 `verify.py` 逐檔比對，全部相符時印出：

```
package: 172 files of data, programs and published results checked, 0 problems
results: 51 files compared with published/
   49 identical
    2 same content
details:
  same content      results_trend/backtest_metadata.json: identical apart from the date and the run time
  same content      results_trend/metadata.json: identical apart from the date and the run time
RESULT: every result reproduces
```

`verify.py` 的判讀方式如下：

- **identical**：重算的檔案與 `published/` 之下的檔案逐位元相同。
- **same content**：內容相同，只有執行日期、執行時間或壓縮檔的時間戳記不同。
- **numerically close**：數值只在末幾位不同（相對差異不超過一億分之一）。換用其他版本的 numpy 或其他處理器時可能出現。
- **figure differs**：圖檔不完全相同，通常是字型或 matplotlib 版本不同；這不影響任何數字。
- **DIFFERENT** 或 **MISSING**：結果沒有重現，程式以錯誤狀態結束，並列出差異。

比對之前，`verify.py` 先依 `CHECKSUMS.sha256` 核對 `data/`、`research/`、`published/` 與 `online_appendix/` 之下的檔案沒有被更動。清單也列了根目錄的檔案；要核對整份清單，可以執行 `sha256sum -c CHECKSUMS.sha256`。

模擬所用的隨機種子都固定在程式內。區間端點與 p 值的模擬誤差，記在各結果檔的 `*_mcse` 欄。

## 程式與結果

| 程式 | 作用 | 結果檔 |
|---|---|---|
| `research/trendlib.py` | 生命表、Poisson Lee–Carter 模型、模擬、分解的共用函式 | |
| `research/trend_backtest.py --intervals` | 疫情前回測 | `results_trend/backtest_*.csv` |
| `research/trend_baseline.py` | 22 個地區的缺口、預測區間與年齡分解 | `results_trend/expected_observed.csv` 等 |
| `research/taiwan_2025.py` | 以內政部原始統計重建臺灣 2025 年的數值 | `results_trend/taiwan_2025_*.csv` |
| `research/trend_age_rates.py` | 年齡組死亡率與趨勢預期值的比 | `results_trend/age_rate_ratios.csv` |
| `research/trend_cohort_check.py` | 其他趨勢基準（不同配適期、年齡—時期—世代模型）及其回測 | `results_trend/cohort_check_*.csv` |
| `research/trend_replicate_dowd.py` | 以已發表研究的設定重算，作為外部檢核 | `results_trend/replication_dowd.csv` |
| `research/taiwan_registration_check.py` | 戶籍人口中死亡以外的變動 | `results_trend/taiwan_registration_check.csv` |
| `research/cause_prepare.py`、`cause_decomposition.py`、`cause_codes.py` | 死因歸類、年齡 × 死因分解、ICD-10 代碼層級的死亡率 | `results_cause/` |
| `research/cause_context.py` | 救護、相驗、死亡場所三項全國統計 | `results_cause/context_*.csv` |
| `research/paper_assets.py` | 由上列結果產生文中的數字、表格與圖，並核對文中以文字敘述的 38 項結果 | `manuscript/generated/`、`manuscript/figures/` |

`manuscript/generated/numbers.json` 收錄可供文中引用的全部數字，`table_cells.json` 是每張表的儲存格。`published/generated/numbers_used.csv` 列出正文與線上附錄實際用到的數字及其值；這個檔在排版時產生，`reproduce.py` 不會重做，僅供對照。

## 表與圖

正文的表（編號依正文中出現的順序）：

- 表 1　疫情前回測：提前 5 年預測的誤差與區間涵蓋率
- 表 2　東亞四地 30 歲平均餘命與趨勢預期值的差距，2020–2025 年
- 表 3　各年齡組死亡率與趨勢預期值的差距（%）
- 表 4　各死因對臺灣 30 歲平均餘命變動的貢獻：2012–2018 年與 2019–2025 年
- 表 5　臺灣 50–64 歲死因別死亡率：2019 年與 2025 年

線上附錄的表：

- 附表 1　22 個地區 2024 年與趨勢預期值的差距
- 附表 2　臺灣各年齡組死亡率與趨勢預期值的差距，2020–2025 年（%）
- 附表 3　不同趨勢基準下臺灣的缺口
- 附表 4　四種基準下臺灣各年齡組對平均餘命缺口的貢獻與死亡率差距
- 附表 5　臺灣各年齡組 2025 年死亡率差距與疫情前回測誤差的比較（對數尺度）
- 附表 6　以 2005–2014 年各年為起點、提前 1、3、5 年的回測結果
- 附表 7　以內政部原始統計重建的數值與 HMD 數值的差
- 附表 8　戶籍人口中死亡以外的淨變動（每百人）
- 附表 9　死因類別與 ICD-10 代碼
- 附表 10　臺灣 50–64 歲死亡率變動最大的 ICD-10 代碼，2019–2025 年
- 附表 11　臺灣各年齡組兩大類死因的死亡率
- 附表 12　正文討論的 ICD-10 代碼歷年死亡人數，2015–2025 年
- 附表 13　不使用死因代碼的三項全國指標，2019–2025 年

圖：

- 圖 1　東亞四地 30 歲平均餘命的實際值與趨勢預期值，2003–2025 年（`published/figures/fig1_trajectories.png`）
- 圖 2　臺灣各年齡組對 30 歲平均餘命缺口的貢獻，2022、2024 與 2025 年（`published/figures/fig2_age_gap.png`）
- 圖 3　臺灣各年齡組的年齡標準化死亡率，2012–2025 年（2019 年 = 1）（`published/figures/fig3_rates_2019.png`）
- 圖 4　臺灣 50–64 歲三大類死因的年齡標準化死亡率，2008–2025 年（`published/figures/fig4_causes_5064.png`）

## 資料與授權

- **HMD**：HMD 團隊建構的資料採 CC BY 4.0 授權。套件只收錄本文用到的死亡數、暴露人年與時期生命表，沒有收錄各國的原始輸入資料。
- **臺灣官方統計**：內政部與衛生福利部的檔案採「政府資料開放授權條款－第1版」。
- 各檔案的來源網址、版本與引用方式見 `DATA_SOURCES.md`。
- **程式碼**：`research/` 之下的程式、`reproduce.py` 與 `verify.py` 採 MIT 授權，條款見 `LICENSE`。著作權人是本文作者，姓名將於論文刊登後補上。
- **結果與線上附錄**：`published/` 與 `online_appendix/` 是該文的結果與附錄，不在 MIT 授權的範圍之內；授權方式將於論文刊登後載明，引用時請註明出處。

## 需要知道的事

- **2025 年的臺灣數值由內政部原始統計重建。** HMD 的臺灣資料到 2024 年為止；`research/taiwan_2025.py` 以內政部按發生日期統計的死亡數與年底人口重建 2025 年，並在 HMD 與內政部資料重疊的年度驗證這個做法（`results_trend/taiwan_2025_bridge_validation.csv`）。
- **HMD 的生命表只用於一項核對。** 本文的生命表指標都由死亡數與暴露人年自行編算。`research/trend_baseline.py` 另由 HMD 公布的時期生命表算出同樣的指標，兩者並列在 `results_trend/observed_measures.csv`（`hmd_official` 與 `difference` 兩欄）；其餘結果都不使用 HMD 的生命表。
- **司法相驗案件數由年報人工讀取。** 這一項數列不是由 `data/` 的檔案算出，而是寫在 `research/cause_context.py` 內；來源見 `DATA_SOURCES.md` 第三節。
- **讀取下載檔的程式以隔離模式（`-I`）執行**，`reproduce.py` 已照此設定。
- **稿件正文與排版程式不在套件內。** 程式註解中提到的 `manuscript/body.tex`、`manuscript/supplement_body.tex`、`research/build_paper.py` 與 `research/check_outputs.py` 屬於排版流程。
- **死因數列在 2019 年不連續**，原因是死因選取準則改版；本文只在同一套準則適用的期間之內比較死因。

## 引用

作者與引用方式將於論文刊登後補上。

## English summary

- **What this is.** Data, code and results for an article (in Chinese) that asks whether adult mortality in Taiwan has returned to its pre-pandemic trend, with a comparison with Japan, South Korea and Hong Kong and an analysis of causes of death. The text of the article is not included; its online appendix is in `online_appendix/`.
- **How to reproduce.** Python 3.12 with the packages in `requirements.txt`; then `python3 reproduce.py` (plain Python; no shell is needed). The script runs the analysis programs in `research/` in order, writes the results to `results_trend/`, `results_cause/` and `manuscript/`, and finally runs `verify.py`, which compares them with the reference results in `published/`.
- **Data.** Deaths, exposures and period life tables for 22 populations from the Human Mortality Database (release of 11 November 2026, CC BY 4.0), and official statistics of Taiwan's Ministry of the Interior and Ministry of Health and Welfare (Open Government Data License, version 1.0). Sources, versions and citations are in `DATA_SOURCES.md` (in Chinese).
- **Licence.** The programs (`research/`, `reproduce.py`, `verify.py`) are released under the MIT License (`LICENSE`). The data keep the licences of their sources. The results in `published/` and the online appendix belong to the article; their licence will be stated when the article is published.
- **Inputs.** The results are recomputed from the raw files in `data/`, with one exception: the yearly numbers of judicial inquests were read by hand from annual reports and are written in `research/cause_context.py`. Figures that the article quotes from official publications or from other studies are not recomputed. The life tables published by HMD are used only to check the life-table measures that the programs compute from deaths and exposures.
