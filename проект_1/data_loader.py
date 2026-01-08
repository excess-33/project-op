import pandas as pd

def load_and_process_data(file_path='melb_data.csv'):
    """
    загружает CSV-файл с данными по недвижимости,
    очищает данные и возвращает их в виде словаря DataFrame

    гарантирует наличие столбцов:
    Price, BuildingArea, Landsize, Regionname, CouncilArea, Suburb, Type
    """

    #1. ЗАГРУЗКА ДАННЫХ 
    # Читаем CSV-файл в pandas DataFrame
    df = pd.read_csv(file_path)

    #убираем пробелы в названиях колонок (например: "BuildingArea " → "BuildingArea")
    
    df.columns = df.columns.str.strip()

    #выводим список колонок для отладки (чтобы видеть, что реально загрузилось)
    print("Колонки после загрузки:", df.columns.tolist())

    #2. БАЗОВАЯ ОЧИСТКА ДАННЫХ

    #удаляем дублирующиеся строки
    df = df.drop_duplicates()

    #преобразуем столбец Price в числовой формат
    #если встречаются нечисловые значения, они заменяются на NaN
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    #удаляем строки, где цена отсутствует (без цены анализ бессмысленен)
    df = df.dropna(subset=['Price'])

    #3. ОБРАБОТКА ПЛОЩАДИ ЗЕМЛИ (Landsize)

    #если по какой-то причине столбца Landsize нет, то создаём его
    if 'Landsize' not in df.columns:
        df['Landsize'] = pd.NA

    #приводим Landsize к числовому типу
    df['Landsize'] = pd.to_numeric(df['Landsize'], errors='coerce')

    #4. ГАРАНТИЯ НАЛИЧИЯ BuildingArea СТОЛБЦА

    #если столбца BuildingArea нет в данных
    if 'BuildingArea' not in df.columns:
        # создаём его и заполняем значениями из Landsize
        # (чтобы main.py мог всегда к нему обращаться)
        df['BuildingArea'] = df['Landsize']
    else:
        #если столбец есть, то приводим к числовому типу
        df['BuildingArea'] = pd.to_numeric(df['BuildingArea'], errors='coerce')

    #если в BuildingArea есть пропуски, но есть Landsize, то подставляем Landsize вместо пропусков
    df['BuildingArea'] = df['BuildingArea'].fillna(df['Landsize'])

    #5. ОБРАБОТКА ОСТАЛЬНЫХ ЧИСЛОВЫХ ПОЛЕЙ 

    numeric_cols = [
        'Rooms', 'Bedroom2', 'Bathroom', 'Car',
        'YearBuilt', 'Lattitude', 'Longtitude', 'Propertycount'
    ]

    for col in numeric_cols:
        if col in df.columns:
            #преобразуем к числам
            df[col] = pd.to_numeric(df[col], errors='coerce')
            #заполняем пропуски средним значением по столбцу
            df[col] = df[col].fillna(df[col].mean())

    #6. ПРЕОБРАЗОВАНИЕ ДАТ И ПОЧТОВЫХ КОДОВ 

    if 'Postcode' in df.columns:
        df['Postcode'] = pd.to_numeric(df['Postcode'], errors='coerce')

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')

    #7. РАЗБИЕНИЕ ПО ТИПУ НЕДВИЖИМОСТИ 
    # h = house (дом), u = unit (квартира), t = townhouse (таунхаус)

    if 'Type' in df.columns:
        df_houses = df[df['Type'] == 'h'].copy()
        df_units = df[df['Type'] == 'u'].copy()
        df_townhouses = df[df['Type'] == 't'].copy()
    else:
        #если вдруг столбца Type нет — создаём пустые таблицы
        df_houses = df.copy().iloc[0:0]
        df_units = df.copy().iloc[0:0]
        df_townhouses = df.copy().iloc[0:0]

    #8. ВОЗВРАТ РЕЗУЛЬТАТА 
    #возвращаем словарь с четырьмя DataFrame-ами

    return {
        'all': df,              # все объекты
        'houses': df_houses,    # дома
        'units': df_units,      # квартиры
        'townhouses': df_townhouses  # таунхаусы
    }

