# 資料來源與授權

套件內的資料檔都是 2026-11-17（臺灣時間）自各來源取得的原檔，未經修改。每個檔案的 SHA-256 列在 `CHECKSUMS.sha256`。

## 一、人類死亡率資料庫（Human Mortality Database，HMD）

- **版本**：2026-11-11 釋出，DOI [10.4054/HMD.Countries.20261111](https://doi.org/10.4054/HMD.Countries.20261111)。
- **位置**：`data/hmd/2026-11-17/raw/core/<地區代碼>/STATS/`。
- **檔案**：每個地區四個檔。`Deaths_1x1.txt` 是按曆年與單一年齡分列的死亡數，`Exposures_1x1.txt` 是暴露人年，`fltper_1x1.txt` 與 `mltper_1x1.txt` 分別是女性與男性的時期生命表。
- **生命表的用途**：本文的生命表指標都由死亡數與暴露人年自行編算，HMD 的生命表只用於核對。`research/trend_baseline.py` 由生命表的 `lx` 與 `Tx` 兩欄算出 30 歲平均餘命與死亡年齡分位數，與自行編算的數值並列在 `results_trend/observed_measures.csv`（`hmd_official` 與 `difference` 兩欄）。文中所述自行編算的 30 歲平均餘命與 HMD 生命表數值的最大差異（約 0.056 年）出自這項比較；其餘結果都不使用 HMD 公布的時期生命表。
- **地區（22 個）**：AUT 奧地利、BEL 比利時、CHE 瑞士、CHL 智利、DEUTNP 德國、DNK 丹麥、EST 愛沙尼亞、FIN 芬蘭、FRATNP 法國、HKG 香港、HRV 克羅埃西亞、JPN 日本、KOR 韓國、LTU 立陶宛、LVA 拉脫維亞、NLD 荷蘭、NOR 挪威、PRT 葡萄牙、SVK 斯洛伐克、SWE 瑞典、TWN 臺灣、USA 美國。
- **授權**：依 HMD 使用協議（<https://www.mortality.org/Data/UserAgreement>，2026-11-16 查閱），HMD 團隊建構的資料採 Creative Commons Attribution 4.0 International（CC BY 4.0）授權。各國統計機關提供的原始輸入資料（Input Data）另依各提供者的授權，本套件沒有收錄。
- **引用**：HMD. Human Mortality Database. Max Planck Institute for Demographic Research (Germany), University of California, Berkeley (USA), and French Institute for Demographic Studies (France). Available at www.mortality.org.
- **請留意**：HMD 會持續修訂資料。本套件所附的是上述版本的副本，用途限於重現本文結果；其他用途請至 <https://www.mortality.org> 取得最新資料。

## 二、臺灣官方統計

下列檔案的授權都是「政府資料開放授權條款－第1版」，使用時須註明出處。

| 檔案（`data/` 之下） | 內容 | 來源 |
|---|---|---|
| `mohw/2026-11-17/downloads/icd_detail/icd_detail.zip` | 衛生福利部統計處「死因統計（國際疾病分類標準之詳細分類）」：全國，年 × 性別 × ICD-10 代碼 × 年齡組（至 100 歲以上），2008–2025 年；另含 1980–2007 年（ICD-9） | <https://www.mohw.gov.tw/dl-83421-2c205514-b970-45c5-9f0c-295569c9c6d5.html>（連結頁 <https://dep.mohw.gov.tw/DOS/cp-6602-74526-113.html>） |
| `mohw/2026-11-17/downloads/condensed_42/all_cause_42.zip` | 衛生福利部統計處「死因統計」開放資料（data.gov.tw 資料集 5965）：鄉鎮市區 × 簡分類 × 性別 × 年齡組，1992–2025 年。本文只用它核對死因歸類 | <https://www.mohw.gov.tw/dl-83417-aa29a17f-289d-4934-a868-fe74b3b14afc.html> |
| `moi/2026-11-17/downloads/deaths_single_age/y3s3-00000.xls` | 內政部戶政司「死亡人口按五齡組（按發生）」，工作表「死亡單齡」：性別 × 單一年齡（至 100 歲以上），1975–2025 年 | <https://www.ris.gov.tw/info-popudata/app/awFastDownload/file/y3s3-00000.xls/y3s3/00000/>（入口 <https://www.ris.gov.tw/app/portal/346>） |
| `moi/2026-11-17/downloads/population_single_age/y1s6-00000.xls` | 內政部戶政司「年底人口按性別及年齡」，工作表「單齡」：1946–2025 年 | <https://www.ris.gov.tw/info-popudata/app/awFastDownload/file/y1s6-00000.xls/y1s6/00000/> |
| `context/2026-11-17/downloads/moi_statis_ems_service_10401_11412/ems_service_monthly.csv` | 內政部統計處內政統計查詢網「消防緊急救護服務─按區域別分」，2015 年 1 月至 2025 年 12 月，含非創傷類與創傷類到院前心肺功能停止人數 | <https://data.gov.tw/dataset/12096> |
| `context/2026-11-17/downloads/moi_place_reg_108/` 至 `moi_place_reg_114/` | 內政部戶政司「人口死亡數按死亡者性別及死亡地點分（按登記）」：鄉鎮市區 × 性別 × 死亡地點，2019–2025 年各一檔 | <https://data.gov.tw/dataset/127529> |

使用這些檔案時需要知道的事：

- **死因選取準則在 2019 年改版。** 衛生福利部自 2019 年起改用 2016 年版 ICD-10 死因選取準則，檔內 2008–2018 年是舊準則之下的原發布數。各死因的數列在 2018 年與 2019 年之間不連續，只能在 2008–2018 年或 2019 年以後各自比較。
- **衛生福利部與內政部的死亡總數略有出入**，每年相差數十至數百人。內政部按發生日期統計的單一年齡死亡數與 HMD 所載相同，30 歲以上各歲相差不到 0.1 人。本文的全死因死亡率一律用 HMD 與內政部資料，衛生福利部資料只取各年齡組之內的死因占比。
- **2022 年起有「司法相驗中」的死亡**（ICD 欄為 ZZZ，每年 124–263 人），本文歸在「其他」。
- **高齡的人口數與 HMD 不同。** HMD 會重新估計 80 歲以上人口，所以 HMD 的暴露人年與內政部年底人口算出的數值有出入：80–89 歲相差不到百分之二，90 歲以上可達數個百分點，100 歲以上更大；哪一方較高，隨年齡與年度而異。各年的比值見 `published/results_trend/taiwan_2025_source_comparison.csv`。`research/taiwan_2025.py` 以最近三個 HMD 年度的比值換算 2025 年的暴露人年。
- 簡分類檔中的 `dead87.csv` 遺失代碼的前導零，程式讀的是 `dead87.txt`。

## 三、沒有收錄、但程式用到其數字的來源

- **司法相驗案件數**：取自法務部法醫研究所「法醫鑑驗業務統計年報」112 年版與 114 年版的表 1-12（<https://www.tpa.moj.gov.tw/292712/292713/680264/292758/Lpsimplelist>）。這些數字由年報人工讀取，寫在 `research/cause_context.py` 內，並與兩個版本的年報核對過；兩個版本合起來涵蓋本文使用的 2019–2025 年。年報本身沒有收錄在套件內，檔案在：
  - 112 年版：<https://www.tpa.moj.gov.tw/media/356151/法醫所-112年法醫鑑驗業務統計年報.pdf>
  - 114 年版：<https://www.tpa.moj.gov.tw/media/427424/法醫所-114年法醫鑑驗業務統計年報.pdf>
