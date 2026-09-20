"""
Shared reconstruction logic for the event-study side of the null-result
check (coefficients #5, #6, #7).

Line-for-line port of CEO_Letter_Event_Study.ipynb's Steps 6.1/14.2 merge
logic (name mappings confirmed against real printed output in-session, not
guessed), starting from three uploaded raw sources:
  - null_check/data/ceo_letter_regression_sample_209.csv (Step 9.5 export:
    the 209-row complete-case CAR/Specificity/Sentiment panel)
  - null_check/data/df_ar_cluster.csv (3-D cluster assignment)
  - null_check/data/Combined_Financials_With_ROA.csv +
    null_check/data/TobinsQ_Results_FullNames.csv (financial controls)
"""

import pandas as pd

REG_SAMPLE_PATH = "null_check/data/ceo_letter_regression_sample_209.csv"
CLUSTER_PATH = "null_check/data/df_ar_cluster.csv"
FIN_PATH = "null_check/data/Combined_Financials_With_ROA.csv"
TOBINSQ_PATH = "null_check/data/TobinsQ_Results_FullNames.csv"

# Verified in-session against CEO_Letter_Event_Study.ipynb's Step 6.1
# printed output (all 22 event-study panel companies map with zero
# unmapped/leftover entries).
SHORT_TO_LONG_NAME = {
    "AIG": "American International Group (AIG)",
    "Chubb": "Chubb",
    "MET": "MetLife, Inc.",
    "MetLife": "MetLife, Inc.",
    "Prudential Financials": "Prudential Financial, Inc.",
    "Prudentials": "Prudential Financial, Inc.",
    "Allstate": "The Allstate Corporation",
    "Progressive": "The Progressive Corporation",
    "Travelers": "The Travelers Companies, Inc.",
    "AXA": "AXA SA",
    "Allianz": "Allianz SE",
    "China Life": "China Life Insurance Company Limited",
    "Daichi": "Dai-ichi Life Holdings",
    "Generali": "Assicurazioni Generali",
    "MSAD": "MS&AD Insurance Group Holdings",
    "Munich RE": "Munich Re",
    "PIC": "People's Insurance Company of China (PICC)",
    "Ping An": "Ping An Insurance Group",
    "Power Corp Canada": "Power Corporation of Canada",
    "Sompo": "Sompo Holdings",
    "Swiss RE": "Swiss Re",
    "Talanx": "Talanx AG",
    "Tokio Marine": "Tokio Marine Holdings",
    "Zurich": "Zurich Insurance Group",
}

# Verified in-session against Step 14.1's printed output (all 22 event-
# study panel companies map with zero unmapped/leftover entries; 'New
# China Life' and 'India Life Insurance' are real Bloomberg-dataset
# companies not part of this 22-firm panel, deliberately mapped to None).
FIN_TO_EVENT_NAME = {
    "AIG": "American International Group (AIG)",
    "AXA": "AXA SA",
    "Allianz": "Allianz SE",
    "Allstate": "The Allstate Corporation",
    "Assicurazioni Generali": "Assicurazioni Generali",
    "China Life": "China Life Insurance Company Limited",
    "Chubb": "Chubb",
    "Daichi": "Dai-ichi Life Holdings",
    "MSAD": "MS&AD Insurance Group Holdings",
    "MetLife": "MetLife, Inc.",
    "Munich RE": "Munich Re",
    "People Insurance China": "People's Insurance Company of China (PICC)",
    "Ping An": "Ping An Insurance Group",
    "Power Corp Canada": "Power Corporation of Canada",
    "Progressive": "The Progressive Corporation",
    "Prudentials": "Prudential Financial, Inc.",
    "Sompo": "Sompo Holdings",
    "Swiss RE": "Swiss Re",
    "Talanx": "Talanx AG",
    "Tokio Marine": "Tokio Marine Holdings",
    "Travelers": "The Travelers Companies, Inc.",
    "Zurich": "Zurich Insurance Group",
    "New China Life": None,
    "India Life Insurance": None,
}


def load_and_merge():
    """
    Reproduces panel14 as built in Step 14.2/14.3: the 209-row event-study
    regression sample, left-merged with the 3-D cluster assignment and
    financial controls (contemporaneous match, Company + Fiscal Year --
    not lagged, per the notebook's explicit design-choice note), with the
    two cluster dummies (Cluster1_HighSpec, Cluster0_HighVolLowSpec;
    Cluster2_Minimal omitted reference) built the same way.
    """
    reg_sample = pd.read_csv(REG_SAMPLE_PATH)
    n_reg = len(reg_sample)

    df_cluster = pd.read_csv(CLUSTER_PATH)
    df_cluster["Company"] = df_cluster["Company Name"].map(SHORT_TO_LONG_NAME)
    unmapped_cluster = df_cluster[df_cluster["Company"].isna()]["Company Name"].unique()
    assert len(unmapped_cluster) == 0, f"Unmapped cluster company names: {sorted(unmapped_cluster)}"
    df_cluster = df_cluster.rename(columns={"Year": "Fiscal Year"})

    df_fin_raw = pd.read_csv(FIN_PATH)
    df_tobinsq_raw = pd.read_csv(TOBINSQ_PATH)
    df_financials_full = pd.merge(df_fin_raw, df_tobinsq_raw, on=["Company", "Year", "Region"], how="inner")

    df_financials_full["Company_event"] = df_financials_full["Company"].map(FIN_TO_EVENT_NAME)
    panel_companies_set = set(SHORT_TO_LONG_NAME.values())
    df_fin_panel = df_financials_full[df_financials_full["Company_event"].isin(panel_companies_set)].copy()
    df_fin_panel = df_fin_panel[[
        "Company_event", "Year", "Leverage", "Size", "ROA", "Revenue Growth YoY (%)"
    ]].rename(columns={
        "Company_event": "Company", "Year": "Fiscal Year", "Revenue Growth YoY (%)": "Revenue_Growth",
    })

    panel14 = reg_sample.merge(
        df_cluster[["Company", "Fiscal Year", "cluster_3d"]], on=["Company", "Fiscal Year"], how="left"
    )
    panel14 = panel14.merge(df_fin_panel, on=["Company", "Fiscal Year"], how="left")
    assert len(panel14) == n_reg, "Row count changed after merge -- duplicate keys."

    panel14["Cluster1_HighSpec"] = (panel14["cluster_3d"] == 1).astype("Int64")
    panel14["Cluster0_HighVolLowSpec"] = (panel14["cluster_3d"] == 0).astype("Int64")
    panel14.loc[panel14["cluster_3d"].isna(), ["Cluster1_HighSpec", "Cluster0_HighVolLowSpec"]] = pd.NA

    meta = {"n_reg_sample": n_reg, "n_missing_cluster": int(panel14["cluster_3d"].isna().sum())}
    return panel14, meta
