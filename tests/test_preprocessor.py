import pytest
import pandas as pd

from preprocessor import (
    BASELINE_FEATURES,
    extract_title,
    normalize_title,
    fill_missing_age,
    fill_missing_cabin,
    fill_missing_embarked,
    fill_missing_fare,
    preprocess,
    preprocess_baseline,
    TITLE_ENCODE,
    DECK_ENCODE,
)


# ---------------------------------------------------------------------------
# extract_title
# ---------------------------------------------------------------------------


def test_extract_title_standard_name():
    assert extract_title("Braund, Mr. Owen Harris") == "Mr"


def test_extract_title_multi_word_title():
    assert extract_title("Rothschild, the Countess. of X") == "the Countess"


def test_extract_title_no_match_returns_unknown():
    assert extract_title("NoCommaNoTitle") == "Unknown"


def test_extract_title_leading_space_stripped():
    assert extract_title("Smith,  Mr. John") == "Mr"


def test_normalize_title_common_and_rare():
    assert normalize_title("Mr") == "Mr"
    assert normalize_title("Mrs") == "Mrs"
    assert normalize_title("Mlle") == "Miss"
    assert normalize_title("Mme") == "Mrs"
    assert normalize_title("Dr") == "Rare"
    assert normalize_title("Col") == "Rare"


# ---------------------------------------------------------------------------
# fill_missing_embarked
# ---------------------------------------------------------------------------


def test_fill_missing_embarked_no_column_unchanged():
    df = pd.DataFrame({"A": [1, 2]})
    result = fill_missing_embarked(df.copy())
    assert list(result.columns) == ["A"]


def test_fill_missing_embarked_fills_from_same_ticket_group():
    df = pd.DataFrame({"Embarked": ["S", "S", None], "Ticket": ["T1", "T1", "T1"]})
    assert fill_missing_embarked(df)["Embarked"].iloc[2] == "S"


def test_fill_missing_embarked_mode_fallback_when_no_ticket_match():
    df = pd.DataFrame(
        {"Embarked": ["S", "S", "C", None], "Ticket": ["T1", "T2", "T3", "T4"]}
    )
    assert fill_missing_embarked(df)["Embarked"].iloc[3] == "S"


# ---------------------------------------------------------------------------
# fill_missing_cabin
# ---------------------------------------------------------------------------


def test_fill_missing_cabin_no_column_unchanged():
    df = pd.DataFrame({"A": [1, 2]})
    assert "Cabin" not in fill_missing_cabin(df.copy()).columns


def test_fill_missing_cabin_fills_from_same_ticket_group():
    df = pd.DataFrame(
        {
            "Cabin": ["C85", None],
            "Ticket": ["T1", "T1"],
            "SibSp": [0, 0],
            "Parch": [0, 0],
        }
    )
    assert fill_missing_cabin(df)["Cabin"].iloc[1] == "C85"


def test_fill_missing_cabin_fills_from_family_group_across_tickets():
    df = pd.DataFrame(
        {
            "Cabin": ["C85", None],
            "Ticket": ["T1", "T2"],
            "SibSp": [1, 1],
            "Parch": [0, 0],
            "Name": ["Smith, Mr. John", "Smith, Mrs. Jane"],
        }
    )
    assert fill_missing_cabin(df)["Cabin"].iloc[1] == "C85"


def test_fill_missing_cabin_unknown_fallback():
    df = pd.DataFrame(
        {
            "Cabin": [None, None],
            "Ticket": ["T1", "T2"],
            "SibSp": [0, 0],
            "Parch": [0, 0],
        }
    )
    assert (fill_missing_cabin(df)["Cabin"] == "Unknown").all()


def test_fill_missing_cabin_no_surname_column_leaked():
    df = pd.DataFrame(
        {
            "Cabin": ["C85", None],
            "Ticket": ["T1", "T2"],
            "SibSp": [1, 1],
            "Parch": [0, 0],
            "Name": ["Smith, Mr. John", "Smith, Mrs. Jane"],
        }
    )
    assert "Surname" not in fill_missing_cabin(df).columns


# ---------------------------------------------------------------------------
# fill_missing_age
# ---------------------------------------------------------------------------


def test_fill_missing_age_fills_by_title_median():
    df = pd.DataFrame(
        {
            "Name": ["Smith, Mr. John", "Doe, Mr. Jake", "Lee, Mr. James"],
            "Age": [30.0, 40.0, None],
            "Pclass": [1, 1, 1],
            "Sex": ["male", "male", "male"],
        }
    )
    assert fill_missing_age(df)["Age"].iloc[2] == 35.0


def test_fill_missing_age_falls_back_to_pclass_sex_median():
    df = pd.DataFrame(
        {
            "Name": ["Smith, Dr. John", "Doe, Mr. Jake", "Lee, Mr. James"],
            "Age": [None, 30.0, 40.0],
            "Pclass": [1, 1, 1],
            "Sex": ["male", "male", "male"],
        }
    )
    assert fill_missing_age(df)["Age"].iloc[0] == 35.0


def test_fill_missing_age_falls_back_to_overall_median():
    df = pd.DataFrame(
        {
            "Name": ["Smith, Dr. John", "Jones, Dr. Jane", "Doe, Mrs. Alice"],
            "Age": [None, None, 50.0],
            "Pclass": [1, 2, 1],
            "Sex": ["male", "female", "female"],
        }
    )
    result = fill_missing_age(df)
    assert result["Age"].iloc[0] == 50.0
    assert result["Age"].iloc[1] == 50.0


def test_fill_missing_age_title_column_dropped():
    df = pd.DataFrame(
        {
            "Name": ["Smith, Mr. John", "Doe, Mr. Jake"],
            "Age": [30.0, None],
            "Pclass": [1, 1],
            "Sex": ["male", "male"],
        }
    )
    assert "Title" not in fill_missing_age(df).columns


def test_fill_missing_age_no_name_column_leaves_data_unchanged():
    df = pd.DataFrame({"Age": [30.0, None]})
    assert fill_missing_age(df.copy())["Age"].isnull().sum() == 1


# ---------------------------------------------------------------------------
# fill_missing_fare
# ---------------------------------------------------------------------------


def test_fill_missing_fare_no_column_unchanged():
    assert "Fare" not in fill_missing_fare(pd.DataFrame({"A": [1]})).columns


def test_fill_missing_fare_fills_from_same_ticket_group():
    df = pd.DataFrame(
        {
            "Fare": [50.0, None],
            "Ticket": ["T1", "T1"],
            "Pclass": [1, 1],
            "Embarked": ["S", "S"],
        }
    )
    assert fill_missing_fare(df)["Fare"].iloc[1] == 50.0


def test_fill_missing_fare_fills_by_pclass_embarked_median():
    df = pd.DataFrame(
        {
            "Fare": [50.0, 60.0, None],
            "Ticket": ["T1", "T2", "T3"],
            "Pclass": [1, 1, 1],
            "Embarked": ["S", "S", "S"],
        }
    )
    assert fill_missing_fare(df)["Fare"].iloc[2] == 55.0


def test_fill_missing_fare_overall_median_fallback():
    df = pd.DataFrame({"Fare": [50.0, None], "Ticket": ["T1", "T2"]})
    assert fill_missing_fare(df)["Fare"].iloc[1] == 50.0


# ---------------------------------------------------------------------------
# preprocess — unit (synthetic single-row data)
# ---------------------------------------------------------------------------


def _minimal_row(**overrides):
    base = {
        "Pclass": [1],
        "Name": ["Smith, Mr. John"],
        "Sex": ["male"],
        "Age": [25.0],
        "SibSp": [0],
        "Parch": [0],
        "Ticket": ["T1"],
        "Fare": [10.0],
        "Cabin": ["C85"],
        "Embarked": ["S"],
    }
    base.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(base)


def test_preprocess_sex_encoded():
    assert preprocess(_minimal_row(Sex="male"))["Sex"].iloc[0] == 0
    assert preprocess(_minimal_row(Sex="female"))["Sex"].iloc[0] == 1


def test_preprocess_embarked_encoded():
    assert preprocess(_minimal_row(Embarked="S"))["Embarked"].iloc[0] == 0
    assert preprocess(_minimal_row(Embarked="C"))["Embarked"].iloc[0] == 1
    assert preprocess(_minimal_row(Embarked="Q"))["Embarked"].iloc[0] == 2


def test_preprocess_familysize():
    assert preprocess(_minimal_row(SibSp=2, Parch=1))["FamilySize"].iloc[0] == 4


def test_preprocess_title_feature():
    assert preprocess(_minimal_row())["Title"].iloc[0] == TITLE_ENCODE["Mr"]
    assert (
        preprocess(_minimal_row(Name="Doe, Mrs. Jane"))["Title"].iloc[0]
        == TITLE_ENCODE["Mrs"]
    )
    assert (
        preprocess(_minimal_row(Name="Doe, Dr. Jane"))["Title"].iloc[0]
        == TITLE_ENCODE["Rare"]
    )


def test_preprocess_has_cabin():
    assert preprocess(_minimal_row(Cabin="C85"))["HasCabin"].iloc[0] == 1
    # NaN cabin gets filled to "Unknown" by fill_missing → HasCabin=0
    df = pd.DataFrame(
        {
            "Pclass": [1, 1],
            "Name": ["A, Mr. X", "B, Mr. Y"],
            "Sex": ["male", "male"],
            "Age": [30.0, 30.0],
            "SibSp": [0, 0],
            "Parch": [0, 0],
            "Ticket": ["T1", "T2"],
            "Fare": [10.0, 10.0],
            "Cabin": ["C85", None],
            "Embarked": ["S", "S"],
        }
    )
    result = preprocess(df)
    assert result["HasCabin"].iloc[0] == 1
    assert result["HasCabin"].iloc[1] == 0


def test_preprocess_cabin_deck():
    assert (
        preprocess(_minimal_row(Cabin="C85"))["CabinDeck"].iloc[0] == DECK_ENCODE["C"]
    )
    assert (
        preprocess(_minimal_row(Cabin="E46"))["CabinDeck"].iloc[0] == DECK_ENCODE["E"]
    )


def test_preprocess_is_alone():
    assert preprocess(_minimal_row(SibSp=0, Parch=0))["IsAlone"].iloc[0] == 1
    assert preprocess(_minimal_row(SibSp=1, Parch=0))["IsAlone"].iloc[0] == 0


def test_preprocess_ticket_group_size_and_fare_per_person():
    df = pd.DataFrame(
        {
            "Pclass": [1, 1],
            "Name": ["A, Mr. X", "B, Mrs. Y"],
            "Sex": ["male", "female"],
            "Age": [30.0, 25.0],
            "SibSp": [1, 1],
            "Parch": [0, 0],
            "Ticket": ["T1", "T1"],
            "Fare": [100.0, 100.0],
            "Cabin": ["C85", "C85"],
            "Embarked": ["S", "S"],
        }
    )
    result = preprocess(df)
    assert (result["TicketGroupSize"] == 2).all()
    assert (result["FarePerPerson"] == 50.0).all()


def test_preprocess_drops_non_predictive_columns():
    result = preprocess(_minimal_row())
    for col in ["Name", "Ticket", "Cabin"]:
        assert col not in result.columns
    assert "Surname" not in result.columns


def test_preprocess_drops_passenger_id_if_present():
    df = _minimal_row()
    df["PassengerId"] = [1]
    assert "PassengerId" not in preprocess(df).columns


def test_preprocess_does_not_modify_input():
    df = _minimal_row()
    original = df.copy(deep=True)
    preprocess(df)
    pd.testing.assert_frame_equal(df, original)


# ---------------------------------------------------------------------------
# preprocess_baseline
# ---------------------------------------------------------------------------


def test_preprocess_baseline_keeps_only_baseline_features_and_target():
    df = _minimal_row()
    df["Survived"] = [1]
    df["Name"] = ["Smith, Mr. John"]
    result = preprocess_baseline(df)

    expected_cols = BASELINE_FEATURES + ["Survived"]
    assert list(result.columns) == expected_cols


def test_preprocess_baseline_applies_basic_encoding_and_imputation():
    df = pd.DataFrame(
        {
            "Pclass": [1, 3],
            "Sex": ["male", "female"],
            "Age": [22.0, None],
            "SibSp": [1, 0],
            "Parch": [0, 0],
            "Fare": [7.25, None],
            "Embarked": ["S", None],
            "Survived": [0, 1],
        }
    )

    result = preprocess_baseline(df)

    assert set(result["Sex"].unique()) <= {0, 1}
    assert set(result["Embarked"].unique()) <= {0, 1, 2}
    assert result.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# preprocess — integration (synthetic multi-row covering all imputation paths)
# ---------------------------------------------------------------------------


def _make_passengers():
    return pd.DataFrame(
        {
            "Pclass": [1, 1, 2, 2, 3, 3, 1, 3],
            "Name": [
                "Smith, Mr. John",
                "Jones, Mrs. Alice",
                "Brown, Miss. Betty",
                "Davis, Mr. James",
                "Wilson, Master. Tom",
                "Moore, Mrs. Carol",
                "Taylor, Dr. Henry",
                "Clark, Mr. Bob",
            ],
            "Sex": [
                "male",
                "female",
                "female",
                "male",
                "male",
                "female",
                "male",
                "male",
            ],
            "Age": [30.0, 25.0, 20.0, None, 10.0, 45.0, None, 35.0],
            "SibSp": [1, 1, 0, 0, 0, 0, 0, 2],
            "Parch": [0, 0, 0, 0, 1, 0, 0, 1],
            "Ticket": ["T1", "T1", "T2", "T3", "T4", "T5", "T6", "T7"],
            "Fare": [100.0, 100.0, 50.0, None, 20.0, 30.0, 200.0, 15.0],
            "Cabin": ["C85", "C85", None, "E46", None, None, None, None],
            "Embarked": ["S", "S", "C", "Q", "S", None, "C", "Q"],
        }
    )


def test_preprocess_no_missing_values_after_imputation():
    processed = preprocess(_make_passengers())
    assert processed[["Age", "Fare", "Embarked"]].isnull().sum().sum() == 0


def test_preprocess_all_encodings_valid():
    processed = preprocess(_make_passengers())
    assert processed["Sex"].isin([0, 1]).all()
    assert processed["Embarked"].isin([0, 1, 2]).all()
    assert processed["Title"].isin(TITLE_ENCODE.values()).all()
    assert processed["CabinDeck"].isin(DECK_ENCODE.values()).all()


def test_preprocess_familysize_formula():
    df = _make_passengers()
    expected = df["SibSp"] + df["Parch"] + 1
    assert (preprocess(df)["FamilySize"] == expected).all()


def test_preprocess_age_title_median_fill():
    df = _make_passengers()
    davis_idx = df[df["Name"] == "Davis, Mr. James"].index[0]
    assert preprocess(df).loc[davis_idx, "Age"] == 32.5


def test_preprocess_age_pclass_sex_fallback_fill():
    df = _make_passengers()
    taylor_idx = df[df["Name"] == "Taylor, Dr. Henry"].index[0]
    assert preprocess(df).loc[taylor_idx, "Age"] == 30.0


def test_preprocess_integration_new_features():
    df = _make_passengers()
    processed = preprocess(df)
    # Smith & Jones share Ticket T1 → TicketGroupSize=2, FarePerPerson=50.0
    assert processed.loc[0, "TicketGroupSize"] == 2
    assert processed.loc[0, "FarePerPerson"] == 50.0
    # Smith row: Cabin="C85" → HasCabin=1, CabinDeck=C=2
    assert processed.loc[0, "HasCabin"] == 1
    assert processed.loc[0, "CabinDeck"] == DECK_ENCODE["C"]
    # Davis row: SibSp=0, Parch=0 → IsAlone=1
    assert processed.loc[3, "IsAlone"] == 1
    # Clark row: SibSp=2, Parch=1 → IsAlone=0
    assert processed.loc[7, "IsAlone"] == 0
