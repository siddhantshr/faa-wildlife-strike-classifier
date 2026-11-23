import pandas as pd
import numpy as np

YEARS = (2010, 2024)
KEEP_COLUMNS = [
    "HEIGHT",
    "SPEED",
    "DISTANCE",
    "INCIDENT_MONTH",
    "AC_CLASS",
    "AC_MASS",
    "NUM_ENGS",
    "TYPE_ENG",
    "PHASE_OF_FLIGHT",
    "TIME_OF_DAY",
    "FAAREGION",
    "SKY",
    "PRECIPITATION",
    "SIZE",
    "NUM_SEEN",
    "NUM_STRUCK",
    "WARNED",
    "SPECIES",
    "DAMAGE_LEVEL"
]
KNOTS_TO_MS = 0.514444
CLEMENTS_PATH = "./data/raw/clements.csv"
AVONET_PATH = "./data/raw/AVONET.xlsx"
manual_name_fixes = {
    "american barn owl": "Tyto alba",
    "western cattle egret": "Bubulcus ibis",
    "northern house wren": "Troglodytes aedon"
}

def clean_names(names: pd.Series) -> pd.Series:
    return (
        names.astype("string")
        .str.lower()
        .str.strip()
        .str.replace(r"[\u2019']", "", regex=True)
        .str.replace("-", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )

def load_common_to_scientific() -> pd.DataFrame:
    clements = pd.read_csv(CLEMENTS_PATH, encoding="latin-1", low_memory=False)
    clements = clements[clements["category"] == "species"]

    df1 = pd.DataFrame({
        "name": clean_names(clements["English name"]),
        "scientific_name": clements["scientific name"]
    })
    df2 = pd.DataFrame({
        "name": list(manual_name_fixes.keys()),
        "scientific_name": list(manual_name_fixes.values())
    })
    return pd.concat([df1, df2], ignore_index=True)

def load_body_mass() -> pd.DataFrame:
    avonet = pd.read_excel(AVONET_PATH, sheet_name="AVONET2_eBird", engine="calamine")

    return pd.DataFrame({
        "scientific_name": avonet["Species2"],
        "BodyMass": avonet["Mass"] / 1000,  # grams -> kg
        "Order": avonet["Order2"],
        "Family": avonet["Family2"],
    })

def add_body_mass(df: pd.DataFrame) -> pd.DataFrame:
    common_to_scientific = load_common_to_scientific()
    body_mass = load_body_mass()

    df = df.assign(name=clean_names(df["SPECIES"]))

    df = df.merge(common_to_scientific, how="left", on="name")
    df = df.merge(body_mass, how="left", on="scientific_name")

    return df.drop(columns="name")

def add_impact_energy(df: pd.DataFrame) -> pd.DataFrame:
    speed = df["SPEED"] * KNOTS_TO_MS
    energy = 0.5 * df["BodyMass"] * speed**2

    return df.assign(
        ImpactEnergy=energy,
        log_ImpactEnergy=np.log1p(energy),
    )

def add_species_group(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(SpeciesGroup=df["Order"].fillna("Unknown"))

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["DAMAGE_LEVEL"].notna()]

    three_class = {"N": 0, "M": 1, "S": 2, "D": 2} # M? has unknown extent, so we keep it as NaN
    binary = {"N": 0, "M": 1, "M?": 1, "S": 1, "D": 1}

    df = df.assign(
        y3=df["DAMAGE_LEVEL"].map(three_class).astype("Int64"),
        y_bin=df["DAMAGE_LEVEL"].map(binary).astype("Int64"),
    )
    return df.drop(columns="DAMAGE_LEVEL")

def load_required_columns() -> pd.DataFrame:
    raw = pd.read_excel("./data/raw/Public.xlsx", engine="calamine")

    raw = raw[
        ((raw["INCIDENT_YEAR"] >= YEARS[0]) & (raw["INCIDENT_YEAR"] <= YEARS[1]))
    ]
    model_b_columns = [
        c for c in raw.columns
        if c.startswith(("STR_", "ING_"))
    ]
    raw = raw[KEEP_COLUMNS + model_b_columns]
    return raw

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_ac_class(df)
    df = clean_type_eng(df)
    df = clean_warned(df)
    df = clean_precipitation(df)
    df = clean_speed(df)

    df = add_body_mass(df)
    df = add_impact_energy(df)
    df = add_species_group(df)
    df = add_targets(df)

    return df.drop(columns=["SPECIES", "scientific_name", "Order", "Family"])

def clean_ac_class(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(AC_CLASS=df["AC_CLASS"].str.strip())
    df = df.assign(
        AC_CLASS=df["AC_CLASS"].map({  # codes from FAA fieldlist.pdf; Z = Unknown -> NaN
            "A": "airplane", "B": "helicopter",
            "C": "other", "D": "other", "F": "other", "I": "other", "J": "other", "Y": "other",
        })
    )
    return df

def clean_type_eng(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(TYPE_ENG=df["TYPE_ENG"].str.strip())
    df = df.assign(
        TYPE_ENG=df["TYPE_ENG"].map({
            "A": "piston",
            "B": "turbojet",
            "C": "turboprop",
            "D": "turbofan",
            "E": "none",
            "F": "turboshaft",
            "Y": "other",
        })
    )
    return df


def clean_warned(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(WARNED=df["WARNED"].str.strip())
    df = df.assign(
        WARNED=df["WARNED"].map({"Yes": "yes", "No": "no"})  # unknown mapped to Nan
    )
    return df

def clean_precipitation(df: pd.DataFrame) -> pd.DataFrame:
    precip = (
        df["PRECIPITATION"].str.lower()
        .str.replace("none", "") # "None, Rain" -> ", rain"
        .str.strip(" ,") # ", rain" -> "rain"
        .replace("", "none")
    )
    precip = precip.mask(precip.str.contains(",", na=False), "mixed")
    precip = precip.mask(precip.isna() & df["SKY"].notna(), "none")
    return df.assign(PRECIPITATION=precip)

def clean_speed(df: pd.DataFrame) -> pd.DataFrame:
    plausible = df["SPEED"].between(0, 400)
    return df.assign(SPEED=df["SPEED"].where(plausible))

def main():
    df = load_required_columns()
    df = engineer_features(df)
    df.to_csv("./data/processed/dataset.csv", index=False)

if __name__ == "__main__":
    main()