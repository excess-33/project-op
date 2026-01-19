import pandas as pd

def load_and_prepare(path: str) -> pd.DataFrame:
    """
    загружает данные и подготавливает их для анализа и визуализации.
    """
    df = pd.read_csv(path)

    # делает даты настоящими объектами datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # колонки, которые должны быть числовыми
    numeric_cols = [
        "Price", "Rooms", "Bedroom2", "Bathroom",
        "Car", "Landsize", "BuildingArea",
        "YearBuilt", "Propertycount"
    ]

    # приведение этих колонок к числовому типу (NaN при ошибке)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # обработка пропусков
    # для числовых колонок заполняем медианой
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # для категориальных колонок заполняем "Unknown"
    cat_cols = ["CouncilArea", "Regionname", "Suburb", "Type"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # дополнительные признаки (в рамках команды)
    if "Date" in df.columns:
        df["SaleYear"] = df["Date"].dt.year
        df["SaleMonth"] = df["Date"].dt.month

    if "YearBuilt" in df.columns:
        df["HouseAge"] = 2026 - df["YearBuilt"]  # возраст дома
        df["IsOldHouse"] = (df["HouseAge"] > 50).astype(int)  # флаг старого дома

    if "Rooms" in df.columns and "Price" in df.columns:
        df["PricePerRoom"] = df["Price"] / df["Rooms"]  # стоимость за комнату

    if "Landsize" in df.columns and "BuildingArea" in df.columns:
        df["BuildRatio"] = df["BuildingArea"] / df["Landsize"]  # коэффициент застройки

    if "Propertycount" in df.columns and "Landsize" in df.columns:
        # плотность застройки: сколько объектов на единицу земли
        df["Density"] = df["Propertycount"] / df["Landsize"].where(df["Landsize"] > 0)

    # ИНДИВИДУАЛЬНАЯ ПРАВКИ СМИРНОВОЙ КСЕНИИ
    # цена за квадратный метр (PricePerM2), с fallback на Landsize
    if "Price" in df.columns and "BuildingArea" in df.columns and "Landsize" in df.columns:
        area = df["BuildingArea"].where(df["BuildingArea"] > 0)
        fallback = df["Landsize"].where(df["Landsize"] > 0)
        ppm2 = df["Price"] / area
        ppm2 = ppm2.where(ppm2.notna(), df["Price"] / fallback)
        df["PricePerM2"] = ppm2

    # категория плотности (низкая/средняя/высокая) для анализа распределения по районам
    if "Density" in df.columns:
        df["DensityCategory"] = pd.qcut(df["Density"], q=3, labels=["Low", "Medium", "High"], duplicates="drop")

    # флаг премиум-недвижимости (на основе цены за м² выше медианы)
    if "PricePerM2" in df.columns:
        median_ppm2 = df["PricePerM2"].median()
        df["IsPremium"] = (df["PricePerM2"] > median_ppm2).astype(int)
        
    # флаг недавней продажи: полезно для анализа трендов по годам
    if "SaleYear" in df.columns:
        df["IsRecentSale"] = (df["SaleYear"] >= 2017).astype(int)
        
    # примерная категория размера жилья по количеству комнат
    if "Rooms" in df.columns:
        df["SizeCategory"] = pd.cut(df["Rooms"], bins=[0, 2, 3, 4, float("inf")], labels=["Small", "Medium", "Large", "Very Large"], right=False)

    return df


def split_by_type(df: pd.DataFrame) -> dict: # функция в рамках командной работы
    """
    разбиение датасета по типу недвижимости.
    возвращает словарь с тремя типами: дом, квартира, таунхаус.
    """
    if "Type" not in df.columns:
        raise ValueError("Column 'Type' not found")

    return {
        "house": df[df["Type"] == "h"].copy(),
        "unit": df[df["Type"] == "u"].copy(),
        "townhouse": df[df["Type"] == "t"].copy()
    }

#НОВЫЕ ФУНКЦИИ В РАМКАХ ИНДИВИДУАЛЬНЫХ ПРАВОК
def get_top_expensive_cheap(df: pd.DataFrame, group_level: str = "Regionname", metric: str = "PricePerM2", top_n: int = 5) -> dict:
    """
    возвращает топ N самых дорогих и самых дешевых предложений по заданной метрике и уровню группировки.
    """
    allowed_levels = {"Regionname", "CouncilArea", "Suburb"}
    if group_level not in allowed_levels:
        raise ValueError(f"Invalid group_level. Allowed: {allowed_levels}")
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in DataFrame")

    grouped = df.groupby(group_level)[metric].agg(["mean", "median"]).reset_index()
    
    # топ дорогих
    top_expensive = grouped.sort_values("median", ascending=False).head(top_n)

    # топ дешевых
    top_cheap = grouped.sort_values("median", ascending=True).head(top_n)
    
    return {
        "top_expensive": top_expensive,
        "top_cheap": top_cheap
    }


def sort_by_density(df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    """
    сортирует датасет по плотности застройки (Density), с опцией по возрастанию/убыванию.
    полезно для анализа перегруженных/свободных районов.
    """
    if "Density" not in df.columns:
        raise ValueError("Column 'Density' not found")
    
    return df.sort_values("Density", ascending=ascending).copy()