"""
🏠 ВИЗУАЛИЗАТОР РЫНКА НЕДВИЖИМОСТИ МЕЛЬБУРНА
Творческая часть: Интерактивные графики и анализ данных

Автор: Подольнев Илья
Задача: Визуализация датасета с объявлениями о продаже недвижимости

ФУНКЦИОНАЛ:
- 7 интерактивных HTML графиков
- Анализ цен по районам
- Гистограммы распределения цен
- Выявление самых дорогих/дешевых предложений
- Анализ по категориям (дом, квартира, апартаменты)
- Корреляционный анализ
- Статистическая сводка
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ThemeConfig:
    """Конфигурация оформления для всех графиков"""
    
    @staticmethod
    def get_plotly_layout():
        """Возвращает единую цветовую схему для всех графиков"""
        return {
            'template': 'plotly_white',
            'font': {'family': 'Arial, sans-serif', 'size': 12, 'color': '#333'},
            'plot_bgcolor': 'rgba(240, 240, 240, 0.5)',
            'paper_bgcolor': 'white',
            'margin': {'l': 60, 'r': 40, 't': 80, 'b': 60},
            'hovermode': 'closest'
        }
    
    @staticmethod
    def get_color_scale(n_colors=10):
        """Возвращает цветовую шкалу: красный=дорого, зелёный=дешево"""
        return px.colors.sequential.RdYlGn_r[:n_colors]


def create_heatmap_prices_by_district(df, price_column='Price', 
                                       district_column='Suburb', 
                                       category_column='Type'):
    """
    Тепловая карта: районы × типы жилья
    Цвет = средняя цена
    
    ПАРАМЕТРЫ:
    - df: DataFrame с данными
    - price_column: название столбца с ценами
    - district_column: название столбца с районами
    - category_column: название столбца с типами жилья
    """
    
    # Группируем данные: средняя цена по районам и типам
    pivot_table = df.groupby([district_column, category_column])[price_column].mean().unstack(fill_value=0)
    
    # Создаём тепловую карту
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='RdYlGn_r',  # Красный=дорого, Зелёный=дешево
        colorbar=dict(title=f"Средняя цена ({price_column})", thickness=15, len=0.7),
        hovertemplate='<b>%{y}</b> - %{x}<br>Средняя цена: $%{z:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='📊 Тепловая карта: Средние цены по районам и типам жилья',
        xaxis_title='Тип жилья',
        yaxis_title='Район (Suburb)',
        height=600,
        **ThemeConfig.get_plotly_layout()
    )
    
    return fig


def create_price_distribution(df, price_column='Price'):
    """
    Распределение цен: гистограмма + KDE кривая
    
    Показывает, как распределены цены на недвижимость
    """
    
    # Удаляем выбросы для лучшей визуализации
    Q1 = df[price_column].quantile(0.25)
    Q3 = df[price_column].quantile(0.75)
    IQR = Q3 - Q1
    clean_data = df[(df[price_column] >= Q1 - 1.5*IQR) & (df[price_column] <= Q3 + 1.5*IQR)]
    
    fig = go.Figure()
    
    # Гистограмма
    fig.add_trace(go.Histogram(
        x=clean_data[price_column],
        nbinsx=50,
        name='Распределение цен',
        marker_color='rgba(33, 128, 141, 0.7)',
        opacity=0.75,
        hovertemplate='Цена: $%{x:,.0f}<br>Количество: %{y}<extra></extra>'
    ))
    
    # Статистика
    mean_price = clean_data[price_column].mean()
    median_price = clean_data[price_column].median()
    
    fig.add_vline(x=mean_price, line_dash='dash', line_color='red', 
                  annotation_text=f'Среднее: ${mean_price:,.0f}')
    fig.add_vline(x=median_price, line_dash='dash', line_color='green',
                  annotation_text=f'Медиана: ${median_price:,.0f}')
    
    fig.update_layout(
        title='📈 Распределение цен на недвижимость',
        xaxis_title='Цена ($)',
        yaxis_title='Количество объявлений',
        barmode='overlay',
        height=500,
        **ThemeConfig.get_plotly_layout()
    )
    
    return fig


def create_box_plot_outliers(df, price_column='Price', category_column='Type'):
    """
    Box plot: анализ выбросов по типам жилья
    
    Показывает квартили, медиану и выбросы для каждой категории
    """
    
    fig = px.box(df, y=price_column, x=category_column, 
                 title='📦 Анализ выбросов цен по типам жилья',
                 labels={price_column: 'Цена ($)', category_column: 'Тип жилья'},
                 color=category_column)
    
    fig.update_layout(
        height=500,
        showlegend=False,
        **ThemeConfig.get_plotly_layout()
    )
    
    return fig


def create_top_bottom_listings(df, price_column='Price', category_column='Type'):
    """
    Топ самых дорогих и дешевых объявлений
    
    Показывает 5 самых дорогих и 5 самых дешевых объявлений
    """
    
    top_5 = df.nlargest(5, price_column)
    bottom_5 = df.nsmallest(5, price_column)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('🏆 Топ 5 самых дорогих', '🏘️ Топ 5 самых дешевых'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Самые дорогие
    fig.add_trace(
        go.Bar(x=top_5[price_column], y=top_5[category_column],
               orientation='h', marker_color='darkred', name='Дорогие',
               hovertemplate='Цена: $%{x:,.0f}<extra></extra>'),
        row=1, col=1
    )
    
    # Самые дешевые
    fig.add_trace(
        go.Bar(x=bottom_5[price_column], y=bottom_5[category_column],
               orientation='h', marker_color='darkgreen', name='Дешевые',
               hovertemplate='Цена: $%{x:,.0f}<extra></extra>'),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text='Цена ($)', row=1, col=1)
    fig.update_xaxes(title_text='Цена ($)', row=1, col=2)
    fig.update_layout(
        title_text='💰 Самые дорогие и дешевые объявления',
        height=500,
        showlegend=False,
        **ThemeConfig.get_plotly_layout()
    )
    
    return fig


def create_comprehensive_dashboard(df, price_column='Price',
                                    district_column='Suburb',
                                    category_column='Type'):
    """
    ГЛАВНЫЙ ДАШБОРД: 4 графика в одном
    
    Включает:
    1. Распределение цен по типам
    2. Средняя цена по районам
    3. Количество объявлений
    4. Статистика
    """
    
    # Подготавливаем данные
    prices_by_category = df.groupby(category_column)[price_column].mean().sort_values(ascending=False)
    prices_by_district = df.groupby(district_column)[price_column].mean().sort_values(ascending=False).head(10)
    count_by_category = df[category_column].value_counts()
    
    # Создаём 4 подграфика
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Средняя цена по типам жилья',
            'Топ-10 районов по цене',
            'Количество объявлений по типам',
            'Статистика по ценам'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'pie'}, {'type': 'table'}]
        ]
    )
    
    # График 1: Средняя цена по типам
    fig.add_trace(
        go.Bar(x=prices_by_category.index, y=prices_by_category.values,
               marker_color='lightblue', name='Средняя цена',
               hovertemplate='%{x}<br>$%{y:,.0f}<extra></extra>'),
        row=1, col=1
    )
    
    # График 2: Топ районов
    fig.add_trace(
        go.Bar(x=prices_by_district.values, y=prices_by_district.index,
               orientation='h', marker_color='lightgreen', name='Цена',
               hovertemplate='$%{x:,.0f}<extra></extra>'),
        row=1, col=2
    )
    
    # График 3: Круговая диаграмма
    fig.add_trace(
        go.Pie(labels=count_by_category.index, values=count_by_category.values,
               name='Количество',
               hovertemplate='%{label}<br>%{value} объявлений<extra></extra>'),
        row=2, col=1
    )
    
    # График 4: Таблица статистики
    stats_data = {
        'Метрика': ['Средняя цена', 'Медиана', 'Минимум', 'Максимум', 'Std Dev'],
        'Значение': [
            f"${df[price_column].mean():,.0f}",
            f"${df[price_column].median():,.0f}",
            f"${df[price_column].min():,.0f}",
            f"${df[price_column].max():,.0f}",
            f"${df[price_column].std():,.0f}"
        ]
    }
    
    fig.add_trace(
        go.Table(
            header=dict(values=['<b>Метрика</b>', '<b>Значение</b>'],
                       fill_color='paleturquoise', align='center'),
            cells=dict(values=[stats_data['Метрика'], stats_data['Значение']],
                      fill_color='lavender', align='left')),
        row=2, col=2
    )
    
    # Обновляем оси
    fig.update_xaxes(title_text='Тип жилья', row=1, col=1)
    fig.update_yaxes(title_text='Цена ($)', row=1, col=1)
    fig.update_xaxes(title_text='Цена ($)', row=1, col=2)
    
    fig.update_layout(
        title_text='📊 ГЛАВНЫЙ ДАШБОРД: Анализ рынка недвижимости Мельбурна',
        height=900,
        showlegend=False,
        **ThemeConfig.get_plotly_layout()
    )
    
    return fig


def create_correlation_matrix(df):
    """
    Матрица корреляций между числовыми признаками
    
    Показывает связи между ценой, площадью, расстоянием и т.д.
    """
    
    # Выбираем только числовые столбцы
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return None, None
    
    # Ограничиваем до 10 столбцов для читаемости
    numeric_cols = numeric_cols[:10]
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        hovertemplate='%{y} vs %{x}<br>Корреляция: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='🔗 Матрица корреляций между показателями',
        height=600,
        **ThemeConfig.get_plotly_layout()
    )
    
    return fig, corr_matrix


def export_analysis_report(df, output_dir='output', price_column='Price',
                            district_column='Suburb', category_column='Type',
                            price_per_sqm_column=None):
    """
    ГЛАВНАЯ ФУНКЦИЯ: создаёт 7 интерактивных HTML файлов
    
    Параметры:
    - df: DataFrame с данными
    - output_dir: папка для сохранения файлов
    - price_column: название столбца с ценами
    - district_column: название столбца с районами
    - category_column: название столбца с типами жилья
    
    СОЗДАЁТ:
    1. 01_main_dashboard.html - Главный дашборд (4 графика)
    2. 02_price_distribution.html - Распределение цен
    3. 03_heatmap_prices.html - Тепловая карта
    4. 04_box_plot.html - Анализ выбросов
    5. 05_top_listings.html - Топ дорогих/дешевых
    6. 06_correlation.html - Корреляции
    7. 07_statistics.html - Статистическая сводка
    """
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("📊 Начинаем создание интерактивных графиков...")
    
    # 1. Главный дашборд
    print("  1️⃣  Создаю главный дашборд...")
    dashboard = create_comprehensive_dashboard(df, price_column, district_column, category_column)
    dashboard.write_html(f'{output_dir}/01_main_dashboard.html')
    
    # 2. Распределение цен
    print("  2️⃣  Создаю распределение цен...")
    price_dist = create_price_distribution(df, price_column)
    price_dist.write_html(f'{output_dir}/02_price_distribution.html')
    
    # 3. Тепловая карта
    print("  3️⃣  Создаю тепловую карту...")
    heatmap = create_heatmap_prices_by_district(df, price_column, district_column, category_column)
    heatmap.write_html(f'{output_dir}/03_heatmap_prices.html')
    
    # 4. Box plot
    print("  4️⃣  Создаю анализ выбросов...")
    boxplot = create_box_plot_outliers(df, price_column, category_column)
    boxplot.write_html(f'{output_dir}/04_box_plot.html')
    
    # 5. Топ объявлений
    print("  5️⃣  Создаю топ дорогих/дешевых...")
    top_bottom = create_top_bottom_listings(df, price_column, category_column)
    top_bottom.write_html(f'{output_dir}/05_top_listings.html')
    
    # 6. Корреляции
    print("  6️⃣  Создаю корреляционный анализ...")
    corr_fig, corr_matrix = create_correlation_matrix(df)
    if corr_fig:
        corr_fig.write_html(f'{output_dir}/06_correlation.html')
    
    # 7. Статистика
    print("  7️⃣  Создаю статистическую сводку...")
    stats_df = pd.DataFrame({
        'Метрика': [
            'Общее количество',
            'Средняя цена',
            'Медиана цены',
            'Минимальная цена',
            'Максимальная цена',
            'Стандартное отклонение',
            'Количество районов',
            'Количество типов жилья'
        ],
        'Значение': [
            len(df),
            f"${df[price_column].mean():,.2f}",
            f"${df[price_column].median():,.2f}",
            f"${df[price_column].min():,.2f}",
            f"${df[price_column].max():,.2f}",
            f"${df[price_column].std():,.2f}",
            df[district_column].nunique(),
            df[category_column].nunique()
        ]
    })
    
    stats_fig = go.Figure(data=[go.Table(
        header=dict(values=['<b>Метрика</b>', '<b>Значение</b>'],
                   fill_color='paleturquoise', align='center', font=dict(size=14)),
        cells=dict(values=[stats_df['Метрика'], stats_df['Значение']],
                  fill_color='lavender', align='left', font=dict(size=12),
                  height=25)
    )])
    
    stats_fig.update_layout(
        title='📋 Статистическая сводка',
        height=400,
        **ThemeConfig.get_plotly_layout()
    )
    stats_fig.write_html(f'{output_dir}/07_statistics.html')
    
    print(f"\n✅ ВСЕ 7 ГРАФИКОВ ГОТОВЫ!")
    print(f"📁 Сохранены в папку: {output_dir}/")
    print("\nФайлы:")
    print("  ✓ 01_main_dashboard.html")
    print("  ✓ 02_price_distribution.html")
    print("  ✓ 03_heatmap_prices.html")
    print("  ✓ 04_box_plot.html")
    print("  ✓ 05_top_listings.html")
    print("  ✓ 06_correlation.html")
    print("  ✓ 07_statistics.html")


if __name__ == "__main__":
    """
    ТОЧКА ВХОДА: запуск анализа
    
    Просто скопируй в Google Colab или запусти в VS Code:
    python visualization_advanced.py
    """
    
    import os
    
    # Создаём папку output
    os.makedirs('output', exist_ok=True)
    
    # Загружаем данные
    df = pd.read_csv('melb_data.csv')
    print(f"✅ Загружено {len(df)} объявлений о недвижимости")
    print(f"📍 Столбцы: {list(df.columns[:10])}")
    
    # Запускаем анализ
    export_analysis_report(
        df,
        output_dir='output',
        price_column='Price',
        district_column='Suburb',
        category_column='Type'
    )
    
    print("\n🎉 ГОТОВО! Открой HTML файлы в браузере!")
