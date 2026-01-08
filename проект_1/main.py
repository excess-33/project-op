
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#загрузка и обработка данных (Ксения) 
from data_loader import load_and_process_data

#загружаем данные правильно
data_dict = load_and_process_data('melb_data.csv')

#основной DataFrame для дальнейшей работы (со всеми объектами)
df = data_dict['all']

#проверка: выводим информацию о загрузке
print("Данные успешно загружены через data_loader!")
print(f"Всего объектов: {len(df)}")
print(f"Домов: {len(data_dict['houses'])}, Квартир: {len(data_dict['units'])}, Таунхаусов: {len(data_dict['townhouses'])}")


#функция для квадратных метров
def add_price_ppm2(df: pd.DataFrame,
                   area_col:str = "BuildingArea",
                   back_area_col:str = "Landsize",
                   out_col:str = "PricePerM2")-> pd.DataFrame:
        d = df.copy()
        area = d[area_col].where(d[area_col] > 0)
        fallback = d[back_area_col].where(d[back_area_col] > 0)
        ppm2 = d["Price"] / area
        ppm2 = ppm2.where(ppm2.notna(), d["Price"] / fallback)
        d[out_col] = ppm2
        return d
#функция для таблицы подсчета через PricePerM2 для разных регионов и тд
def summarize_market(df: pd.DataFrame,
                     group_level: str = "Regionname",
                     metric: str = "Price_per_m2",
                     top_n: int = 15) -> pd.DataFrame:
    allowed = {"Regionname", "CouncilArea", "Suburb"}
    if group_level not in allowed:
        raise ValueError(...)
    if metric not in df.columns:
        raise ValueError(...)
    d = df.dropna(subset=[group_level, metric]).copy()
    out = (d.groupby(group_level)[metric]
             .agg(count="count",
                  mean="mean",
                  median="median",
                  min="min",
                  max="max")
             .sort_values("median", ascending=False))
    if top_n:
        out = out.head(top_n)
    return out.reset_index()
df2 = add_price_ppm2(df)
#можно выбирать по чему делать расчет
summarize_market(df2, group_level="Regionname", metric="PricePerM2", top_n=None)
summarize_market(df2, group_level="CouncilArea", metric="PricePerM2", top_n=20)
summarize_market(df2, group_level="Suburb", metric="PricePerM2", top_n=20)
print(df2)
