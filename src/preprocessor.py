import pandas as pd
import re


TITLE_MAP = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
}

TITLE_ENCODE = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}

DECK_ENCODE = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "T": 7,
    "U": 8,
}


def extract_title(name):
    match = re.search(r",\s*([^\.]+)\.", name)
    if match:
        return match.group(1).strip()
    return "Unknown"


def normalize_title(title):
    return TITLE_MAP.get(title, "Rare")


def fill_missing_embarked(data: pd.DataFrame) -> pd.DataFrame:
    if "Embarked" in data.columns and data["Embarked"].isnull().any():
        # Try to fill missing Embarked by ticket group
        ticket_groups = data.groupby("Ticket")
        for ticket, group in ticket_groups:
            known_embarked = group["Embarked"].dropna().unique()
            if len(known_embarked) == 1:
                idx_missing = group[group["Embarked"].isnull()].index
                data.loc[idx_missing, "Embarked"] = known_embarked[0]
        # Fill any remaining with mode
        data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
    return data


def fill_missing_cabin(data: pd.DataFrame) -> pd.DataFrame:
    if "Cabin" in data.columns and data["Cabin"].isnull().any():
        # Try to fill missing Cabin by ticket group
        ticket_groups = data.groupby("Ticket")
        for ticket, group in ticket_groups:
            known_cabin = group["Cabin"].dropna().unique()
            if len(known_cabin) == 1:
                idx_missing = group[group["Cabin"].isnull()].index
                data.loc[idx_missing, "Cabin"] = known_cabin[0]

        # Try to fill by surname+SibSp+Parch group if still missing
        def surname(name):
            return name.split(",")[0].strip() if pd.notnull(name) else ""

        if "Name" in data.columns:
            data["Surname"] = data["Name"].apply(surname)
            fam_groups = data.groupby(["Surname", "SibSp", "Parch"])
            for _, group in fam_groups:
                known_cabin = group["Cabin"].dropna().unique()
                if len(known_cabin) == 1:
                    idx_missing = group[group["Cabin"].isnull()].index
                    data.loc[idx_missing, "Cabin"] = known_cabin[0]
            data.drop(columns=["Surname"], inplace=True)
        # Fill any remaining with 'Unknown'
        data["Cabin"] = data["Cabin"].fillna("Unknown")
    return data


def fill_missing_age(data: pd.DataFrame) -> pd.DataFrame:
    if "Name" in data.columns and "Age" in data.columns:
        data["Title"] = data["Name"].apply(extract_title)
        if data["Age"].isnull().any():
            # Median by Title
            age_group_median = data.groupby("Title")["Age"].transform("median")
            data["Age"] = data["Age"].fillna(age_group_median)
        if data["Age"].isnull().any() and all(
            col in data.columns for col in ["Pclass", "Sex"]
        ):
            # Median by Pclass and Sex
            age_group_median = data.groupby(["Pclass", "Sex"])["Age"].transform(
                "median"
            )
            data["Age"] = data["Age"].fillna(age_group_median)
        if data["Age"].isnull().any():
            # Fill any remaining with overall median
            data["Age"] = data["Age"].fillna(data["Age"].median())
        data.drop(columns=["Title"], inplace=True)
    return data


def fill_missing_fare(data: pd.DataFrame) -> pd.DataFrame:
    if "Fare" in data.columns and data["Fare"].isnull().any():
        # Try to fill missing Fare by ticket group
        ticket_groups = data.groupby("Ticket")
        for ticket, group in ticket_groups:
            known_fare = group["Fare"].dropna().unique()
            if len(known_fare) == 1:
                idx_missing = group[group["Fare"].isnull()].index
                data.loc[idx_missing, "Fare"] = known_fare[0]
        # Fill by (Pclass, Embarked) group median if still missing
        if all(col in data.columns for col in ["Pclass", "Embarked"]):
            fare_group_median = data.groupby(["Pclass", "Embarked"])["Fare"].transform(
                "median"
            )
            data["Fare"] = data["Fare"].fillna(fare_group_median)
        # Fill any remaining with overall median
        data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    return data


def fill_missing(data: pd.DataFrame) -> pd.DataFrame:
    data = fill_missing_embarked(data)
    data = fill_missing_cabin(data)
    data = fill_missing_age(data)
    data = fill_missing_fare(data)
    return data


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()  # Work on a copy to avoid changing the original

    data = fill_missing(data)

    data["Title"] = data["Name"].apply(extract_title).apply(normalize_title)
    data["Title"] = data["Title"].map(TITLE_ENCODE)

    data["HasCabin"] = (data["Cabin"] != "Unknown").astype(int)
    data["CabinDeck"] = (
        data["Cabin"].str[0].map(DECK_ENCODE).fillna(DECK_ENCODE["U"]).astype(int)
    )

    data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
    data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

    ticket_counts = data["Ticket"].value_counts()
    data["TicketGroupSize"] = data["Ticket"].map(ticket_counts)
    data["FarePerPerson"] = data["Fare"] / data["TicketGroupSize"]

    data.drop(
        columns=["PassengerId", "Name", "Ticket", "Cabin"],
        errors="ignore",
        inplace=True,
    )

    return data


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data("train.csv")
    processed = preprocess(df)
    print(processed.info())
    print(processed.head())
