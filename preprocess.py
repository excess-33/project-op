import pandas as pd

def load_and_prepare(path: str) -> pd.DataFrame:
    """
    Загружает данные и подготавливает их для анализа и визуализации.
    """
    df = pd.read_csv(path)

    
    
     #делает даты настоящими объектами datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    #колонки, которые должны быть числовыми
    numeric_cols = [
        "Price", "Rooms", "Bedroom2", "Bathroom",
        "Car", "Landsize", "BuildingArea",
        "YearBuilt", "Propertycount"
    ]

    #приведение этих колонок к числовому типу (NaN при ошибке)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    
    #обработка пропусков 
    #для числовых колонок заполняем медианой
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    #для категориальных колонок заполняем "Unknown"
    cat_cols = ["CouncilArea", "Regionname", "Suburb", "Type"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    #дополнительные признаки 
    if "Date" in df.columns:
        df["SaleYear"] = df["Date"].dt.year
        df["SaleMonth"] = df["Date"].dt.month

    if "YearBuilt" in df.columns:
        df["HouseAge"] = 2026 - df["YearBuilt"]  #возраст дома
        df["IsOldHouse"] = (df["HouseAge"] > 50).astype(int)  #флаг старого дома

    if "Rooms" in df.columns and "Price" in df.columns:
        df["PricePerRoom"] = df["Price"] / df["Rooms"]  #стоимость за комнату

    if "Landsize" in df.columns and "BuildingArea" in df.columns:
        df["BuildRatio"] = df["BuildingArea"] / df["Landsize"]  #коэффициент застройки

    if "Propertycount" in df.columns and "Landsize" in df.columns:
        #плотность застройки: сколько объектов на единицу земли
        df["Density"] = df["Propertycount"] / df["Landsize"].where(df["Landsize"] > 0)

    return df


def split_by_type(df: pd.DataFrame) -> dict:
    """
    Разбиение датасета по типу недвижимости.
    Возвращает словарь с тремя типами: дом, квартира, таунхаус.
    """
    if "Type" not in df.columns:
        raise ValueError("Column 'Type' not found")

    return {
        "house": df[df["Type"] == "h"].copy(),
        "unit": df[df["Type"] == "u"].copy(),
        "townhouse": df[df["Type"] == "t"].copy()
    }
