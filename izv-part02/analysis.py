#!/usr/bin/env python3
# coding=utf-8
"""Analysis of the Czech Police traffic accident statistics (2023-2026).

The module loads the yearly HTML-based XLS files from a ZIP archive, cleans
them and draws three figures: causes of accidents by vehicle type and region,
persons injured or killed by age group and year, and monthly accident counts
by visibility in four selected regions.
"""

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import zipfile

# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz

AI_USAGE = """
AI - DeepSeek-V4.1-Flash-1017 byla použita k pomáhání s debuggováním,
hledání drobných syntaktických chyb a revizí kódu.
"""

REGIONS = {
    0: "PHA", 1: "STC", 2: "JHC", 3: "PLK", 4: "ULK", 5: "HKK",
    6: "JHM", 7: "MSK", 14: "OLK", 15: "ZLK", 16: "VYS",
    17: "PAK", 18: "LBK", 19: "KVK",
}
VEHICLE_TYPES = {2: "Motorcycle", 3: "Car", 5: "Truck", 6: "Truck", 7: "Truck"}
CAUSES = {
    1: "Not caused by driver", 2: "Inadequate speed",
    3: "Wrong overtaking", 4: "Failure to yield",
    5: "Wrong driving style", 6: "Technical defect",
}
OUTCOMES = {1: "Death", 2: "Severe injury", 3: "Light injury"}
AGE_GROUPS = ["0–17", "18–24", "25–44", "45–64", "65+"]
VISIBILITY_ORDER = [
    "Day - clear", "Day - reduced", "Night - lit", "Night - unlit"
]
VISIBILITY = {
    1: VISIBILITY_ORDER[0], 2: VISIBILITY_ORDER[1], 3: VISIBILITY_ORDER[1],
    4: VISIBILITY_ORDER[2], 5: VISIBILITY_ORDER[2],
    6: VISIBILITY_ORDER[3], 7: VISIBILITY_ORDER[3],
}
# teple barvy pro den, studene pro noc
VISIBILITY_COLORS = dict(zip(
    VISIBILITY_ORDER, ["#f2a900", "#b35806", "#6baed6", "#08306b"]
))

RAW_COLUMNS = ["p1", "p2a", "p4a", "p12", "p19"]
KEEP = ["p1", "p12", "p19", "date", "region"]

# Ukol 1: nacteni dat ze ZIP souboru


def load_data(filename: str, ds: str) -> pd.DataFrame:
    """Load and combine one dataset from yearly HTML files in a ZIP archive."""
    with zipfile.ZipFile(filename) as archive:
        frames = [
            pd.read_html(
                archive.open(f"{year}/I{ds}.xls"),
                encoding="cp1250",
            )[0]
            for year in range(2023, 2027)
        ]

    data = pd.concat(frames, ignore_index=True)
    unnamed = data.columns.astype(str).str.match(r"^Unnamed")
    return data.loc[:, ~(unnamed | data.isna().all())]

# Ukol 2: zpracovani dat


def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Return a cleaned copy with parsed dates and readable regions."""
    """
        naive version, copies all collumns instead of only needed ones 
        data = df.copy()
    """
    data = df[RAW_COLUMNS].copy()
    data["date"] = pd.to_datetime(data["p2a"], dayfirst=True, errors="coerce")
    data["region"] = (
        pd.to_numeric(data["p4a"], errors="coerce").map(REGIONS)
        .astype("category")
    )
    data = data.drop_duplicates(subset="p1", keep="first")[KEEP]

    if verbose:
        size_mb = data.memory_usage(deep=True, index=True).sum() / 1_000_000
        print(f"new_size={size_mb:.1f} MB")

    return data

# Ukol 3: hlavni priciny nehod podle druhu vozidla


def _prepare_causes(df: pd.DataFrame, df_vehicles: pd.DataFrame
                    ) -> pd.DataFrame:
    """Count unique accidents by cause, region and vehicle type."""
    vehicles = df_vehicles[["p1", "p44"]].copy()
    vehicles["vehicle_type"] = pd.to_numeric(
        vehicles["p44"], errors="coerce"
    ).map(VEHICLE_TYPES)
    accidents = df[["p1", "p12", "region"]].drop_duplicates("p1")
    merged = accidents.merge(vehicles[["p1", "vehicle_type"]], on="p1")
    merged["cause"] = (
        pd.to_numeric(merged["p12"], errors="coerce") // 100
    ).map(CAUSES)
    merged = merged.dropna(subset=["cause", "vehicle_type", "region"])
    return (
        merged.groupby(["cause", "region", "vehicle_type"], observed=True)["p1"]
        .nunique().rename("accidents").reset_index()
    )


def plot_causes(df: pd.DataFrame, df_vehicles: pd.DataFrame,
                fig_location: str = None, show_figure: bool = False):
    """Plot unique accidents by cause, region and vehicle type."""
    counts = _prepare_causes(df, df_vehicles)
    type_order = ["Motorcycle", "Car", "Truck"]
    grid = sns.catplot(
        data=counts, x="region", y="accidents", hue="vehicle_type",
        hue_order=type_order, order=sorted(REGIONS.values()),
        col="cause", col_order=list(CAUSES.values()), col_wrap=2,
        kind="bar", palette="Set2", sharex=False, sharey=False,
        height=2.4, aspect=1.5,
    )
    grid.set(yscale="log")
    grid.set_titles("{col_name}")
    grid.set_xticklabels(rotation=90, fontsize=7)
    _label_axes(grid, "Region", "Unique accidents (log scale)")
    grid.legend.set_title("Vehicle type")
    grid.figure.suptitle("Main causes of accidents by vehicle type", y=1.02)
    _finish_plot(grid, fig_location, show_figure)

# Ukol 4: vek osob a nasledky v jednotlivych letech


def _prepare_age(df: pd.DataFrame, df_consequences: pd.DataFrame
                 ) -> pd.DataFrame:
    """Count persons by outcome, age group and accident year."""
    accidents = df[["p1", "date"]].drop_duplicates("p1")
    people = accidents.merge(df_consequences[["p1", "p59g", "p59d"]], on="p1")
    people["year"] = people["date"].dt.year
    people["outcome"] = pd.to_numeric(
        people["p59g"], errors="coerce"
    ).map(OUTCOMES)
    people["age"] = pd.to_numeric(people["p59d"], errors="coerce")
    people = people[
        people["year"].isin([2023, 2024, 2025])
        & people["age"].ge(0)
        & people["outcome"].notna()
    ].copy()
    people["age_group"] = pd.cut(
        people["age"], bins=[-1, 17, 24, 44, 64, float("inf")],
        labels=AGE_GROUPS, right=True,
    )
    return (
        people.groupby(["outcome", "age_group", "year"], observed=True)
        .size().rename("persons").reset_index()
    )


def plot_age(df: pd.DataFrame, df_consequences: pd.DataFrame,
             fig_location: str = None, show_figure: bool = False):
    """Plot person counts by age group, consequence, and accident year."""
    counts = _prepare_age(df, df_consequences)
    grid = sns.catplot(
        data=counts, x="age_group", y="persons", hue="year",
        col="outcome", col_order=list(OUTCOMES.values()), kind="bar",
        order=AGE_GROUPS, palette="viridis", sharey=False,
        height=3, aspect=0.75,
    )
    grid.set_titles("{col_name}")
    grid.set_xticklabels(rotation=45, fontsize=8)
    _label_axes(grid, "Age group", "Persons")
    grid.legend.set_title("Year")
    grid.figure.suptitle("Injured and killed persons by age group", y=1.03)
    _finish_plot(grid, fig_location, show_figure)

# Ukol 5: viditelnost (den / noc) v case


def _monthly_visibility(df: pd.DataFrame, regions: list) -> pd.DataFrame:
    """Return monthly accident counts per region and visibility category."""
    data = df.loc[
        df["region"].isin(regions), ["p1", "date", "region", "p19"]
    ].copy()
    data["visibility"] = pd.to_numeric(
        data["p19"], errors="coerce"
    ).map(VISIBILITY)
    data = data.dropna(subset=["date", "visibility"])
    daily = data.pivot_table(
        index=["region", "date"], columns="visibility", values="p1",
        aggfunc="nunique", fill_value=0,
    ).reindex(columns=VISIBILITY_ORDER, fill_value=0)
    return (
        daily.groupby(level="region").resample("MS", level="date").sum()
        .reset_index()
        .melt(
            id_vars=["region", "date"], var_name="visibility",
            value_name="accidents",
        )
    )


def plot_visibility(df: pd.DataFrame, fig_location: str = None,
                    show_figure: bool = False):
    """Plot monthly accident counts by visibility in four selected regions."""
    regions = ["PHA", "STC", "JHM", "MSK"]
    grid = sns.relplot(
        data=_monthly_visibility(df, regions), x="date", y="accidents",
        hue="visibility", hue_order=VISIBILITY_ORDER,
        palette=VISIBILITY_COLORS, col="region", col_order=regions,
        col_wrap=2, kind="line", height=2.6, aspect=1.3,
    )
    grid.set_titles("Region: {col_name}")
    for axis in grid.axes.flat:
        axis.set_xlim(pd.Timestamp("2023-01-01"), pd.Timestamp("2026-01-01"))
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
        axis.tick_params(axis="x", labelrotation=45)
    _label_axes(grid, "Month", "Accidents")
    grid.legend.set_title("Visibility")
    grid.figure.suptitle("Monthly accidents by visibility", y=1.02)
    _finish_plot(grid, fig_location, show_figure)


def _label_axes(grid, xlabel: str, ylabel: str) -> None:
    """Set axis labels on every subplot of a Seaborn grid."""
    for axis in grid.axes.flat:
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)


def _finish_plot(grid, fig_location: str, show_figure: bool) -> None:
    """Save and/or show a Seaborn figure, then close it."""
    grid.tight_layout()
    if fig_location:
        grid.figure.savefig(fig_location, bbox_inches="tight")
    if show_figure:
        plt.show()
    plt.close(grid.figure)


if __name__ == "__main__":
    # zde je ukazka pouziti, tuto cast muzete modifikovat podle libosti
    # skript nebude pri testovani pousten primo, ale budou volany konkreni
    # funkce.

    df = load_data("data_23_26.zip", "nehody")
    df_consequences = load_data("data_23_26.zip", "nasledky")
    df_vehicles = load_data("data_23_26.zip", "Vozidla")
    df2 = parse_data(df, True)

    plot_causes(df2, df_vehicles, "01_causes.png")
    plot_age(df2, df_consequences, "02_age.png", False)
    plot_visibility(df2, "03_visibility.png")

# Poznamka:
# pro to, abyste se vyhnuli castemu nacitani muzete vyuzit napr
# VS Code a oznaceni jako bunky (radek #%%% )
# Pak muzete data jednou nacist a dale ladit jednotlive funkce
# Pripadne si muzete vysledny dataframe ulozit nekam na disk (pro ladici
# ucely) a nacitat jej naparsovany z disku
