#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ геохимических данных скважин с использованием pairwise log-ratios (PLR)
Отбор проб и признаков для минимизации дисперсии внутри скважин
Только для скважин (field исключены из анализа)
"""

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("tab10")

# Параметры
MIN_PROBES_RATIO = 0.70  # Минимум 70% проб должно остаться
MIN_FEATURES = 10        # Минимум признаков PLR
EPSILON = 1e-10          # Малое число для избежания деления на ноль

def load_data(filepath):
    """Загрузка данных из файла"""
    df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
    print(f"Загружено {df.shape[0]} проб, {df.shape[1]} столбцов")
    return df

def separate_wells_and_field(df, well_col_idx=2):
    """Разделение на скважины и полевые точки"""
    well_col = df.columns[well_col_idx]
    wells_df = df[~df[well_col].astype(str).str.contains('field', na=False)].copy()
    field_df = df[df[well_col].astype(str).str.contains('field', na=False)].copy()
    print(f"Скважины: {wells_df.shape[0]} проб, Полевые точки: {field_df.shape[0]} проб")
    return wells_df, field_df

def get_hydrocarbon_columns(df, start_idx=3):
    """Получение имен столбцов с углеводородами (пропускаем первые 3)"""
    return df.columns[start_idx:]

def compute_pairwise_logratios(df, hc_columns):
    """Вычисление всех pairwise log-ratios"""
    n_features = len(hc_columns)
    n_pairs = n_features * (n_features - 1) // 2
    
    print(f"Создание {n_pairs} pairwise log-ratios из {n_features} признаков...")
    
    plr_data = {}
    pair_names = []
    
    for i, j in combinations(range(n_features), 2):
        col_i = hc_columns[i]
        col_j = hc_columns[j]
        
        # Избегаем деления на ноль и логарифма от нуля
        vals_i = np.clip(df[col_i].values, EPSILON, None)
        vals_j = np.clip(df[col_j].values, EPSILON, None)
        
        plr = np.log(vals_i / vals_j)
        pair_name = f"ln({col_i}/{col_j})"
        
        plr_data[pair_name] = plr
        pair_names.append((col_i, col_j))
    
    plr_df = pd.DataFrame(plr_data)
    
    # Сохраняем метаданные
    metadata_cols = df.columns[:3].tolist()
    plr_df = pd.concat([df[metadata_cols], plr_df], axis=1)
    
    print(f"Создано {len(pair_names)} PLR признаков")
    return plr_df, pair_names

def select_probes_by_variance(plr_df, well_col, stage_col, min_ratio=MIN_PROBES_RATIO):
    """
    Отбор проб методом итеративного удаления наиболее удалённых проб
    Только для скважин (field исключаются из процесса)
    """
    wells = plr_df[well_col].unique()
    
    # Исключаем field из процесса отбора
    wells = [w for w in wells if 'field' not in str(w).lower()]
    
    plr_columns = plr_df.columns[3:]  # PLR признаки
    
    selected_indices = []
    removal_log = []
    
    print(f"\nОтбор проб для {len(wells)} скважин...")
    
    for well in wells:
        well_mask = plr_df[well_col] == well
        well_indices = plr_df[well_mask].index.tolist()
        n_original = len(well_indices)
        n_min = int(n_original * min_ratio)
        
        print(f"  Скважина {well}: {n_original} проб, минимум {n_min}")
        
        current_indices = well_indices.copy()
        
        # Итеративное удаление
        while len(current_indices) > n_min:
            # Вычисляем матрицу расстояний
            well_data = plr_df.loc[current_indices, plr_columns]
            
            # Стандартизация
            scaler = StandardScaler()
            well_data_scaled = scaler.fit_transform(well_data)
            
            # Попарные евклидовы расстояния
            n = len(current_indices)
            distances = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.sqrt(np.sum((well_data_scaled[i] - well_data_scaled[j])**2))
                    distances[i, j] = dist
                    distances[j, i] = dist
            
            # Среднее расстояние для каждой пробы
            mean_distances = distances.mean(axis=1)
            
            # Находим пробу с максимальным средним расстоянием
            max_idx = np.argmax(mean_distances)
            probe_to_remove = current_indices[max_idx]
            
            # Логирование
            removal_log.append({
                'well': well,
                'probe_id': plr_df.loc[probe_to_remove, plr_df.columns[0]],
                'stage': plr_df.loc[probe_to_remove, stage_col],
                'mean_distance': mean_distances[max_idx],
                'remaining_probes': len(current_indices) - 1
            })
            
            # Удаляем
            current_indices.remove(probe_to_remove)
        
        selected_indices.extend(current_indices)
        print(f"    Осталось: {len(current_indices)} проб ({len(current_indices)/n_original*100:.1f}%)")
    
    plr_selected = plr_df.loc[selected_indices].copy()
    
    return plr_selected, removal_log

def select_features_by_variance(plr_df, plr_columns, min_features=MIN_FEATURES):
    """Отбор признаков с наименьшей дисперсией"""
    
    variances = plr_df[plr_columns].var().sort_values()
    
    # Выбираем min_features признаков с наименьшей дисперсией
    n_to_select = max(min_features, len(variances) // 2)  # Хотя бы половина или min_features
    selected_features = variances.head(n_to_select).index.tolist()
    
    print(f"\nОтбор признаков: {len(selected_features)} из {len(plr_columns)} (минимум {min_features})")
    
    return selected_features, variances

def visualize_results(original_plr, selected_plr, selected_features, variances, 
                     well_col, stage_col, removal_log, output_prefix='results'):
    """Визуализация результатов"""
    
    plr_columns = original_plr.columns[3:]
    
    # Исключаем field из всех графиков
    orig_wells_only = original_plr[~original_plr[well_col].astype(str).str.contains('field', na=False)].copy()
    sel_wells_only = selected_plr[~selected_plr[well_col].astype(str).str.contains('field', na=False)].copy()
    
    # Получаем уникальные этапы для цветовой карты
    stages = sorted(orig_wells_only[stage_col].unique())
    n_stages = len(stages)
    stage_colors = plt.cm.tab10(np.linspace(0, 1, min(n_stages, 10)))
    stage_color_map = {stage: stage_colors[i % len(stage_colors)] for i, stage in enumerate(stages)}
    
    fig_scale = 1.5
    
    # 1. Тепловая карта попарных расстояний (до и после) для КАЖДОЙ скважины
    print("\nСоздание тепловых карт по скважинам...")
    
    wells = sorted(orig_wells_only[well_col].unique())
    
    for well in wells:
        orig_well = orig_wells_only[orig_wells_only[well_col] == well]
        sel_well = sel_wells_only[sel_wells_only[well_col] == well]
        
        if len(orig_well) < 2 or len(sel_well) < 2:
            continue
        
        # Только выбранные признаки
        orig_data = orig_well[selected_features]
        sel_data = sel_well[selected_features]
        
        # Стандартизация
        scaler = StandardScaler()
        orig_scaled = scaler.fit_transform(orig_data)
        sel_scaled = scaler.fit_transform(sel_data)
        
        # Матрицы расстояний
        def compute_distance_matrix(data):
            n = data.shape[0]
            dist = np.zeros((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    d = np.sqrt(np.sum((data[i] - data[j])**2))
                    dist[i, j] = d
                    dist[j, i] = d
            return dist
        
        orig_dist = compute_distance_matrix(orig_scaled)
        sel_dist = compute_distance_matrix(sel_scaled)
        
        fig, axes = plt.subplots(1, 2, figsize=(12*fig_scale, 5*fig_scale))
        
        im0 = axes[0].imshow(orig_dist, cmap='viridis', aspect='auto')
        axes[0].set_title(f'До отбора\n{well}: {orig_dist.shape[0]} проб', fontsize=12)
        axes[0].set_xlabel('Пробы')
        axes[0].set_ylabel('Пробы')
        plt.colorbar(im0, ax=axes[0], label='Расстояние')
        
        im1 = axes[1].imshow(sel_dist, cmap='viridis', aspect='auto')
        axes[1].set_title(f'После отбора\n{well}: {sel_dist.shape[0]} проб', fontsize=12)
        axes[1].set_xlabel('Пробы')
        axes[1].set_ylabel('Пробы')
        plt.colorbar(im1, ax=axes[1], label='Расстояние')
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_distance_{well}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Сохранено: {output_prefix}_distance_{well}.png")
    
    # 2. PCA визуализация для КАЖДОЙ скважины отдельно
    print("\nСоздание PCA проекций по скважинам...")
    
    for well in wells:
        orig_well = orig_wells_only[orig_wells_only[well_col] == well].copy()
        sel_well = sel_wells_only[sel_wells_only[well_col] == well].copy()
        
        if len(orig_well) < 2:
            continue
        
        # PCA fit на исходных данных скважины
        pca = PCA(n_components=2)
        orig_pca = pca.fit_transform(orig_well[selected_features])
        
        if len(sel_well) >= 2:
            sel_pca = pca.transform(sel_well[selected_features])
        else:
            sel_pca = None
        
        # Создаем фигуру с двумя подграфиками
        fig, axes = plt.subplots(1, 2, figsize=(14*fig_scale, 6*fig_scale))
        
        # До отбора - раскрашиваем по этапам
        for stage in orig_well[stage_col].unique():
            mask = orig_well[stage_col] == stage
            color = stage_color_map.get(stage, 'gray')
            axes[0].scatter(orig_pca[mask, 0], orig_pca[mask, 1], 
                           c=[color], label=f'Этап {stage}', 
                           alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
        
        axes[0].set_title(f'PCA: До отбора\n{well} ({len(orig_well)} проб)', fontsize=14)
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        axes[0].legend(loc='best', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # После отбора
        if sel_pca is not None and len(sel_well) >= 2:
            for stage in sel_well[stage_col].unique():
                mask = sel_well[stage_col] == stage
                color = stage_color_map.get(stage, 'gray')
                axes[1].scatter(sel_pca[mask, 0], sel_pca[mask, 1], 
                               c=[color], label=f'Этап {stage}', 
                               alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
            
            axes[1].set_title(f'PCA: После отбора\n{well} ({len(sel_well)} проб)', fontsize=14)
            axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
            axes[1].legend(loc='best', fontsize=10)
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'Нет данных\nдля отображения', 
                        transform=axes[1].transAxes, ha='center', va='center', fontsize=14)
            axes[1].set_title(f'PCA: После отбора\n{well}', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_pca_{well}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Сохранено: {output_prefix}_pca_{well}.png")
    
    # 3. Общая PCA для всех скважин (для сравнения)
    print("\nСоздание общей PCA проекции...")
    
    pca_all = PCA(n_components=2)
    orig_pca_all = pca_all.fit_transform(orig_wells_only[selected_features])
    sel_pca_all = pca_all.transform(sel_wells_only[selected_features])
    
    fig, axes = plt.subplots(1, 2, figsize=(16*fig_scale, 6*fig_scale))
    
    # До отбора - по скважинам
    unique_wells = orig_wells_only[well_col].unique()
    for i, well in enumerate(unique_wells):
        mask = orig_wells_only[well_col] == well
        axes[0].scatter(orig_pca_all[mask, 0], orig_pca_all[mask, 1], 
                       label=well, alpha=0.6, s=60, edgecolors='black', linewidth=0.3)
    
    axes[0].set_title(f'PCA: До отбора (все скважины, {len(orig_wells_only)} проб)', fontsize=14)
    axes[0].set_xlabel(f'PC1 ({pca_all.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca_all.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # После отбора - по скважинам
    for i, well in enumerate(sel_wells_only[well_col].unique()):
        mask = sel_wells_only[well_col] == well
        axes[1].scatter(sel_pca_all[mask, 0], sel_pca_all[mask, 1], 
                       label=well, alpha=0.6, s=60, edgecolors='black', linewidth=0.3)
    
    axes[1].set_title(f'PCA: После отбора (все скважины, {len(sel_wells_only)} проб)', fontsize=14)
    axes[1].set_xlabel(f'PC1 ({pca_all.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca_all.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_pca_all_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Сохранено: {output_prefix}_pca_all_comparison.png")
    
    # 4. PCA по этапам внутри каждой скважины
    print("\nСоздание PCA проекций по этапам для каждой скважины...")
    
    for well in wells:
        orig_well = orig_wells_only[orig_wells_only[well_col] == well].copy()
        sel_well = sel_wells_only[sel_wells_only[well_col] == well].copy()
        
        if len(orig_well) < 2:
            continue
        
        pca = PCA(n_components=2)
        orig_pca = pca.fit_transform(orig_well[selected_features])
        
        fig, ax = plt.subplots(figsize=(10*fig_scale, 8*fig_scale))
        
        # Раскрашиваем по этапам
        for stage in sorted(orig_well[stage_col].unique()):
            mask = orig_well[stage_col] == stage
            color = stage_color_map.get(stage, 'gray')
            ax.scatter(orig_pca[mask, 0], orig_pca[mask, 1], 
                      c=[color], label=f'Этап {stage}', 
                      alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
        
        ax.set_title(f'PCA: {well}\nРаскраска по этапам ({len(orig_well)} проб)', fontsize=14)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_pca_stages_{well}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Сохранено: {output_prefix}_pca_stages_{well}.png")
    
    # 5. Распределение дисперсий признаков
    print("\nСоздание графиков дисперсий...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14*fig_scale, 5*fig_scale))
    
    # Все дисперсии
    axes[0].hist(variances.values, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(variances[selected_features].max(), color='red', linestyle='--', 
                    label=f'Макс дисперсия отобранных: {variances[selected_features].max():.3f}')
    axes[0].set_title(f'Распределение дисперсий всех {len(variances)} PLR признаков', fontsize=12)
    axes[0].set_xlabel('Дисперсия')
    axes[0].set_ylabel('Частота')
    axes[0].legend()
    
    # Дисперсии отобранных
    sel_variances = variances[selected_features]
    axes[1].bar(range(len(sel_variances)), sel_variances.sort_values().values, 
                edgecolor='black', alpha=0.7)
    axes[1].set_title(f'Дисперсии {len(selected_features)} отобранных PLR признаков', fontsize=12)
    axes[1].set_xlabel('Признак (ранжирован)')
    axes[1].set_ylabel('Дисперсия')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_variance_distribution.png', dpi=150)
    plt.close()
    print(f"  Сохранено: {output_prefix}_variance_distribution.png")
    
    # 6. Дисперсия по скважинам и этапам
    print("\nАнализ дисперсии по скважинам и этапам...")
    
    well_variances = sel_wells_only.groupby(well_col)[selected_features].var().mean(axis=1)
    stage_variances = sel_wells_only.groupby(stage_col)[selected_features].var().mean(axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12*fig_scale, 5*fig_scale))
    
    well_variances.sort_values().plot(kind='barh', ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_title('Средняя дисперсия по скважинам', fontsize=12)
    axes[0].set_xlabel('Дисперсия')
    axes[0].set_ylabel('Скважина')
    
    stage_variances.sort_values().plot(kind='barh', ax=axes[1], color='coral', edgecolor='black')
    axes[1].set_title('Средняя дисперсия по этапам', fontsize=12)
    axes[1].set_xlabel('Дисперсия')
    axes[1].set_ylabel('Этап')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_variance_by_groups.png', dpi=150)
    plt.close()
    print(f"  Сохранено: {output_prefix}_variance_by_groups.png")
    
    # 7. График удаления проб
    if removal_log:
        removal_df = pd.DataFrame(removal_log)
        
        fig, ax = plt.subplots(figsize=(10*fig_scale, 5*fig_scale))
        
        for well in removal_df['well'].unique():
            well_data = removal_df[removal_df['well'] == well]
            ax.plot(well_data['remaining_probes'], well_data['mean_distance'], 
                   marker='o', label=well, linewidth=2)
        
        ax.set_title('Процесс удаления проб: среднее расстояние vs количество проб', fontsize=12)
        ax.set_xlabel('Количество оставшихся проб')
        ax.set_ylabel('Среднее попарное расстояние удаляемой пробы')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_probe_removal.png', dpi=150)
        plt.close()
        print(f"  Сохранено: {output_prefix}_probe_removal.png")

def main():
    """Основная функция"""
    print("="*60)
    print("Анализ геохимических данных с pairwise log-ratios")
    print("="*60)
    
    # 1. Загрузка данных
    df = load_data('Composite_data.txt')
    
    # 2. Разделение на скважины и field
    wells_df, field_df = separate_wells_and_field(df)
    
    # 3. Получение имён столбцов с углеводородами
    hc_columns = get_hydrocarbon_columns(df)
    print(f"Углеводородные признаки: {len(hc_columns)}")
    
    # 4. Вычисление pairwise log-ratios для всех данных
    plr_df, pair_names = compute_pairwise_logratios(df, hc_columns)
    
    # 5. Отбор проб (только для скважин)
    well_col = df.columns[2]  # Столбец с метками скважин
    stage_col = df.columns[1]  # Столбец с этапами
    
    plr_selected, removal_log = select_probes_by_variance(
        plr_df, well_col, stage_col,
        min_ratio=MIN_PROBES_RATIO
    )
    
    # 6. Отбор признаков по дисперсии
    plr_columns = plr_selected.columns[3:]
    selected_features, variances = select_features_by_variance(
        plr_selected, plr_columns, 
        min_features=MIN_FEATURES
    )
    
    # Финальный датасет
    final_df = plr_selected[[well_col, stage_col, df.columns[0]] + selected_features]
    
    print(f"\n{'='*60}")
    print("ИТОГИ:")
    print(f"{'='*60}")
    print(f"Исходно проб: {df.shape[0]}")
    print(f"Осталось проб: {final_df.shape[0]} ({final_df.shape[0]/df.shape[0]*100:.1f}%)")
    print(f"Исходно PLR признаков: {len(plr_columns)}")
    print(f"Осталось PLR признаков: {len(selected_features)}")
    print(f"{'='*60}")
    
    # 7. Визуализация
    visualize_results(
        plr_df, plr_selected, selected_features, variances,
        well_col, stage_col, removal_log,
        output_prefix='results_plr'
    )
    
    # 8. Сохранение результатов
    final_df.to_csv('results_selected_probes_plr.csv', index=False)
    print(f"\nСохранено: results_selected_probes_plr.csv")
    
    # Сохранение информации об отобранных признаках
    features_df = pd.DataFrame({
        'PLR_feature': selected_features,
        'variance': variances[selected_features].values,
        'component_1': [name.split('/')[0].replace('ln(', '') for name in selected_features],
        'component_2': [name.split('/')[1].replace(')', '') for name in selected_features]
    })
    features_df.to_csv('results_selected_features_plr.csv', index=False)
    print(f"Сохранено: results_selected_features_plr.csv")
    
    # Отчёт
    with open('results_report_plr.txt', 'w', encoding='utf-8') as f:
        f.write("ОТЧЁТ ПО АНАЛИЗУ GEOХИМИЧЕСКИХ ДАННЫХ\n")
        f.write("="*60 + "\n\n")
        f.write(f"Исходный файл: Composite_data.txt\n")
        f.write(f"Исходно проб: {df.shape[0]}\n")
        f.write(f"Осталось проб: {final_df.shape[0]} ({final_df.shape[0]/df.shape[0]*100:.1f}%)\n")
        f.write(f"Минимальный порог сохранения проб: {MIN_PROBES_RATIO*100}%\n\n")
        
        f.write(f"Исходно углеводородных признаков: {len(hc_columns)}\n")
        f.write(f"Создано PLR признаков: {len(plr_columns)}\n")
        f.write(f"Осталось PLR признаков: {len(selected_features)}\n")
        f.write(f"Минимальное количество признаков: {MIN_FEATURES}\n\n")
        
        f.write("ОТОБРАННЫЕ PLR ПРИЗНАКИ:\n")
        f.write("-"*40 + "\n")
        for feat in selected_features[:20]:
            f.write(f"  {feat}: {variances[feat]:.4f}\n")
        if len(selected_features) > 20:
            f.write(f"  ... и ещё {len(selected_features)-20}\n")
        
        f.write("\n\nДИСПЕРСИИ ПО СКВАЖИНАМ:\n")
        f.write("-"*40 + "\n")
        well_var = plr_selected.groupby(well_col)[selected_features].var().mean(axis=1)
        for well, var in well_var.sort_values().items():
            f.write(f"  {well}: {var:.4f}\n")
        
        f.write("\n\nДИСПЕРСИИ ПО ЭТАПАМ:\n")
        f.write("-"*40 + "\n")
        stage_var = plr_selected.groupby(stage_col)[selected_features].var().mean(axis=1)
        for stage, var in stage_var.sort_values().items():
            f.write(f"  {stage}: {var:.4f}\n")
    
    print(f"Сохранено: results_report_plr.txt")
    
    print(f"\n{'='*60}")
    print("АНАЛИЗ ЗАВЕРШЁН УСПЕШНО!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
