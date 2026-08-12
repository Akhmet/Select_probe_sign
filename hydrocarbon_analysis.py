#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ и отбор проб углеводородов с минимизацией дисперсии между этапами
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gmean
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля визуализации
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class HydrocarbonSampleSelector:
    """
    Класс для отбора проб и признаков с минимизацией дисперсии между этапами
    """
    
    def __init__(self, min_samples_ratio=0.7, min_features=10):
        """
        Параметры:
        -----------
        min_samples_ratio : float
            Минимальная доля сохраняемых проб (по умолчанию 0.7 = 70%)
        min_features : int
            Минимальное количество log-отношений признаков
        """
        self.min_samples_ratio = min_samples_ratio
        self.min_features = min_features
        self.data_raw = None
        self.data_clr = None
        self.selected_data = None
        self.selected_features = None
        self.well_labels = None
        self.stage_labels = None
        self.sample_ids = None
        self.results_summary = {}
        
    def load_data(self, filepath):
        """Загрузка данных из файла"""
        print(f"Загрузка данных из {filepath}...")
        self.data_raw = pd.read_csv(filepath, sep='\t')
        print(f"Загружено {len(self.data_raw)} проб с {len(self.data_raw.columns) - 3} признаками углеводородов")
        return self
    
    def preprocess_data(self):
        """Предобработка данных и разделение на мета-данные и концентрации"""
        # Извлечение мета-данных
        self.sample_ids = self.data_raw.iloc[:, 0]
        self.stage_labels = self.data_raw.iloc[:, 1]
        self.well_labels = self.data_raw.iloc[:, 2]
        
        # Извлечение концентраций
        hydrocarbon_cols = self.data_raw.columns[3:]
        concentrations = self.data_raw[hydrocarbon_cols].values
        
        # Замена нулей на малое положительное число для логарифмирования
        min_positive = concentrations[concentrations > 0].min()
        concentrations = np.where(concentrations == 0, min_positive * 0.01, concentrations)
        
        self.hydrocarbon_names = hydrocarbon_cols.tolist()
        self.concentrations = concentrations
        
        print(f"Уникальные скважины: {np.unique(self.well_labels)}")
        print(f"Уникальные этапы: {np.unique(self.stage_labels)}")
        return self
    
    def clr_transform(self):
        """
        Центрированное лог-отношение (Centered Log-Ratio transformation)
        CLR(x_i) = ln(x_i / g(x)), где g(x) - геометрическое среднее
        """
        print("Применение CLR трансформации...")
        
        # Вычисление геометрического среднего для каждой пробы
        geo_means = np.apply_along_axis(gmean, 1, self.concentrations)
        
        # CLR трансформация
        self.data_clr = np.log(self.concentrations / geo_means[:, np.newaxis])
        
        # Создание DataFrame с CLR данными
        self.clr_df = pd.DataFrame(
            self.data_clr,
            columns=[f"CLR_{col}" for col in self.hydrocarbon_names],
            index=self.sample_ids
        )
        
        print(f"CLR трансформация выполнена. Размерность: {self.data_clr.shape}")
        return self
    
    def compute_pairwise_distances(self, data=None):
        """Вычисление попарных евклидовых расстояний между пробами"""
        if data is None:
            data = self.data_clr
        distances = squareform(pdist(data, metric='euclidean'))
        return distances
    
    def select_samples_by_variance(self, well_mask, stage_col):
        """
        Отбор проб с минимизацией дисперсии внутри скважины
        Метод: итеративное удаление проб с наибольшим вкладом в дисперсию
        """
        well_data = self.data_clr[well_mask]
        well_stages = stage_col[well_mask]
        well_ids = self.sample_ids[well_mask]
        
        n_samples = len(well_data)
        min_samples = max(int(n_samples * self.min_samples_ratio), 1)
        
        if n_samples <= min_samples:
            return well_ids.values, np.arange(len(self.hydrocarbon_names))
        
        current_indices = list(range(n_samples))
        current_data = well_data.copy()
        
        # Итеративное удаление проб
        while len(current_indices) > min_samples:
            # Вычисление дисперсии по каждому признаку
            variance_per_feature = np.var(current_data, axis=0)
            total_variance = np.sum(variance_per_feature)
            
            # Вычисление вклада каждой пробы в общую дисперсию
            sample_contributions = []
            for i in range(len(current_indices)):
                # Дисперсия без текущей пробы
                remaining_data = np.delete(current_data, i, axis=0)
                if len(remaining_data) > 1:
                    remaining_variance = np.sum(np.var(remaining_data, axis=0))
                    contribution = total_variance - remaining_variance
                else:
                    contribution = 0
                sample_contributions.append(contribution)
            
            # Удаление пробы с максимальным вкладом в дисперсию
            max_contrib_idx = np.argmax(sample_contributions)
            current_indices.pop(max_contrib_idx)
            current_data = np.delete(current_data, max_contrib_idx, axis=0)
        
        selected_ids = well_ids.iloc[current_indices].values
        feature_indices = np.arange(len(self.hydrocarbon_names))
        
        return selected_ids, feature_indices
    
    def select_features_by_variance(self, well_mask, selected_sample_ids):
        """
        Отбор признаков (log-отношений) с наименьшей дисперсией
        """
        # Получаем данные только для отобранных проб этой скважины
        selected_mask = np.isin(self.sample_ids, selected_sample_ids)
        combined_mask = well_mask & selected_mask
        well_data = self.data_clr[combined_mask]
        
        # Вычисление дисперсии по каждому признаку
        variances = np.var(well_data, axis=0)
        
        # Сортировка признаков по дисперсии
        sorted_indices = np.argsort(variances)
        
        # Выбор минимального количества признаков или всех если мало
        n_features_to_select = max(self.min_features, len(sorted_indices))
        selected_indices = sorted_indices[:n_features_to_select]
        
        return selected_indices
    
    def fit(self):
        """
        Основной метод отбора проб и признаков для каждой скважины
        """
        print("\n" + "="*60)
        print("НАЧАЛО ОТБОРА ПРОБ И ПРИЗНАКОВ")
        print("="*60)
        
        unique_wells = np.unique(self.well_labels)
        
        all_selected_sample_ids = []
        all_selected_features = set(range(len(self.hydrocarbon_names)))
        selection_details = {}
        
        for well in unique_wells:
            print(f"\n--- Обработка скважины: {well} ---")
            
            well_mask = self.well_labels == well
            n_original = np.sum(well_mask)
            
            # Отбор проб
            selected_sample_ids, _ = self.select_samples_by_variance(
                well_mask, self.stage_labels
            )
            
            # Маска отобранных проб
            selected_sample_mask = np.isin(self.sample_ids, selected_sample_ids)
            
            # Отбор признаков
            selected_features = self.select_features_by_variance(
                well_mask, selected_sample_ids
            )
            
            # Обновление общего набора признаков (пересечение)
            all_selected_features = all_selected_features.intersection(set(selected_features))
            
            all_selected_sample_ids.extend(selected_sample_ids)
            
            n_selected = len(selected_sample_ids)
            retention_rate = n_selected / n_original * 100
            
            selection_details[well] = {
                'original': n_original,
                'selected': n_selected,
                'retention_rate': retention_rate,
                'selected_features': len(selected_features)
            }
            
            print(f"  Оригинально проб: {n_original}")
            print(f"  Отобрано проб: {n_selected} ({retention_rate:.1f}%)")
            print(f"  Отобрано признаков: {len(selected_features)}")
        
        # Финальный отбор проб и признаков
        final_sample_mask = np.isin(self.sample_ids, all_selected_sample_ids)
        final_features = list(all_selected_features)
        
        self.selected_data = self.data_clr[final_sample_mask][:, final_features]
        self.selected_features = [self.hydrocarbon_names[i] for i in final_features]
        
        self.selected_sample_ids = self.sample_ids[final_sample_mask]
        self.selected_well_labels = self.well_labels[final_sample_mask]
        self.selected_stage_labels = self.stage_labels[final_sample_mask]
        
        # Сохранение результатов
        self.results_summary = {
            'total_samples_original': len(self.sample_ids),
            'total_samples_selected': len(self.selected_sample_ids),
            'sample_retention_rate': len(self.selected_sample_ids) / len(self.sample_ids) * 100,
            'total_features_original': len(self.hydrocarbon_names),
            'total_features_selected': len(final_features),
            'well_details': selection_details
        }
        
        print("\n" + "="*60)
        print("ЗАВЕРШЕНИЕ ОТБОРА")
        print("="*60)
        print(f"Всего проб оригинально: {self.results_summary['total_samples_original']}")
        print(f"Всего проб отобрано: {self.results_summary['total_samples_selected']}")
        print(f"Процент сохранения проб: {self.results_summary['sample_retention_rate']:.1f}%")
        print(f"Всего признаков оригинально: {self.results_summary['total_features_original']}")
        print(f"Всего признаков отобрано: {self.results_summary['total_features_selected']}")
        
        return self
    
    def compute_variance_by_stage(self):
        """Вычисление дисперсии по этапам для каждой скважины"""
        variance_results = {}
        
        unique_wells = np.unique(self.selected_well_labels)
        unique_stages = np.unique(self.selected_stage_labels)
        
        for well in unique_wells:
            well_mask = self.selected_well_labels == well
            well_data = self.selected_data[well_mask]
            well_stages = self.selected_stage_labels[well_mask]
            
            variance_results[well] = {}
            for stage in unique_stages:
                stage_mask = well_stages == stage
                if np.sum(stage_mask) > 1:
                    stage_variance = np.var(well_data[stage_mask], axis=0).mean()
                    variance_results[well][stage] = stage_variance
                else:
                    variance_results[well][stage] = 0
        
        return variance_results
    
    def visualize_similarity(self, save_path_prefix='results'):
        """Визуализация подобия проб"""
        print("\nСоздание визуализаций подобия проб...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # 1. Тепловая карта попарных расстояний (до отбора - по всем данным)
        ax1 = axes[0, 0]
        distances_full = self.compute_pairwise_distances(self.data_clr)
        im1 = ax1.imshow(distances_full, cmap='viridis', aspect='auto')
        ax1.set_title('Попарные расстояния (все пробы)', fontsize=12)
        ax1.set_xlabel('Пробы')
        ax1.set_ylabel('Пробы')
        plt.colorbar(im1, ax=ax1, label='Евклидово расстояние')
        
        # 2. Тепловая карта попарных расстояний (после отбора)
        ax2 = axes[0, 1]
        distances_selected = self.compute_pairwise_distances(self.selected_data)
        im2 = ax2.imshow(distances_selected, cmap='viridis', aspect='auto')
        ax2.set_title('Попарные расстояния (отобранные пробы)', fontsize=12)
        ax2.set_xlabel('Пробы')
        ax2.set_ylabel('Пробы')
        plt.colorbar(im2, ax=ax2, label='Евклидово расстояние')
        
        # 3. PCA визуализация
        ax3 = axes[1, 0]
        if len(self.selected_data) > 2:
            pca = PCA(n_components=min(2, len(self.selected_data)-1))
            pca_result = pca.fit_transform(self.selected_data)
            
            scatter = ax3.scatter(pca_result[:, 0], pca_result[:, 1], 
                                 c=pd.factorize(self.selected_well_labels)[0], 
                                 cmap='tab10', alpha=0.7, s=50)
            ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            ax3.set_title('PCA проекция отобранных проб')
            ax3.grid(True, alpha=0.3)
        
        # 4. t-SNE визуализация
        ax4 = axes[1, 1]
        if len(self.selected_data) > 3:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.selected_data)-1))
            tsne_result = tsne.fit_transform(self.selected_data)
            
            ax4.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                       c=pd.factorize(self.selected_stage_labels)[0], 
                       cmap='Set2', alpha=0.7, s=50)
            ax4.set_xlabel('t-SNE 1')
            ax4.set_ylabel('t-SNE 2')
            ax4.set_title('t-SNE проекция (раскраска по этапам)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path_prefix}_similarity.png', dpi=300, bbox_inches='tight')
        print(f"Сохранено: {save_path_prefix}_similarity.png")
        plt.show()
        
        return self
    
    def visualize_variance(self, save_path_prefix='results'):
        """Визуализация дисперсии"""
        print("\nСоздание визуализаций дисперсии...")
        
        variance_results = self.compute_variance_by_stage()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Box plot дисперсий по признакам (до и после)
        ax1 = axes[0, 0]
        variances_before = np.var(self.data_clr, axis=0)
        variances_after = np.var(self.selected_data, axis=0)
        
        bp_data = [variances_before, variances_after]
        bp = ax1.boxplot(bp_data, labels=['До отбора', 'После отбора'], patch_artist=True)
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax1.set_ylabel('Дисперсия')
        ax1.set_title('Распределение дисперсий по признакам')
        ax1.grid(True, alpha=0.3)
        
        # 2. Дисперсия по скважинам и этапам
        ax2 = axes[0, 1]
        wells = list(variance_results.keys())
        stages = list(set().union(*[dict(v).keys() for v in variance_results.values()]))
        
        variance_matrix = []
        for well in wells:
            row = [variance_results[well].get(stage, 0) for stage in stages]
            variance_matrix.append(row)
        
        im2 = ax2.imshow(variance_matrix, cmap='YlOrRd', aspect='auto')
        ax2.set_yticks(range(len(wells)))
        ax2.set_yticklabels(wells)
        ax2.set_xticks(range(len(stages)))
        ax2.set_xticklabels(stages, rotation=45)
        ax2.set_title('Средняя дисперсия по скважинам и этапам')
        plt.colorbar(im2, ax=ax2, label='Дисперсия')
        
        # 3. Heatmap дисперсий признаков
        ax3 = axes[1, 0]
        if len(self.selected_features) <= 20:
            feature_variances = np.var(self.selected_data, axis=0)
            bars = ax3.barh(range(len(feature_variances)), feature_variances)
            ax3.set_yticks(range(len(self.selected_features)))
            ax3.set_yticklabels([f[:15] for f in self.selected_features], fontsize=8)
            ax3.set_xlabel('Дисперсия')
            ax3.set_title('Дисперсия отобранных признаков')
            ax3.invert_yaxis()
        else:
            # Если признаков много, показать топ-20
            top_20_idx = np.argsort(np.var(self.selected_data, axis=0))[-20:]
            feature_variances = np.var(self.selected_data, axis=0)[top_20_idx]
            feature_names = [self.selected_features[i][:15] for i in top_20_idx]
            bars = ax3.barh(range(len(feature_variances)), feature_variances)
            ax3.set_yticks(range(len(feature_variances)))
            ax3.set_yticklabels(feature_names, fontsize=8)
            ax3.set_xlabel('Дисперсия')
            ax3.set_title('Топ-20 отобранных признаков по дисперсии')
            ax3.invert_yaxis()
        
        # 4. График сохранения проб по скважинам
        ax4 = axes[1, 1]
        well_names = list(self.results_summary['well_details'].keys())
        original_counts = [self.results_summary['well_details'][w]['original'] for w in well_names]
        selected_counts = [self.results_summary['well_details'][w]['selected'] for w in well_names]
        
        x = np.arange(len(well_names))
        width = 0.35
        
        ax4.bar(x - width/2, original_counts, width, label='Оригинально', color='lightcoral')
        ax4.bar(x + width/2, selected_counts, width, label='Отобрано', color='lightgreen')
        ax4.set_ylabel('Количество проб')
        ax4.set_title('Количество проб по скважинам (до и после отбора)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(well_names, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{save_path_prefix}_variance.png', dpi=300, bbox_inches='tight')
        print(f"Сохранено: {save_path_prefix}_variance.png")
        plt.show()
        
        return self
    
    def save_results(self, output_prefix='results'):
        """Сохранение результатов"""
        print("\nСохранение результатов...")
        
        # Сохранение отобранных данных
        selected_df = pd.DataFrame(
            self.selected_data,
            columns=self.selected_features,
            index=self.selected_sample_ids
        )
        selected_df.insert(0, 'Stage', self.selected_stage_labels)
        selected_df.insert(0, 'Well', self.selected_well_labels)
        selected_df.to_csv(f'{output_prefix}_selected_data.csv')
        print(f"Сохранено: {output_prefix}_selected_data.csv")
        
        # Сохранение списка признаков
        features_df = pd.DataFrame({'Feature': self.selected_features})
        features_df.to_csv(f'{output_prefix}_selected_features.csv', index=False)
        print(f"Сохранено: {output_prefix}_selected_features.csv")
        
        # Сохранение сводки
        summary_df = pd.DataFrame({
            'Metric': [
                'Original Samples',
                'Selected Samples',
                'Sample Retention Rate (%)',
                'Original Features',
                'Selected Features'
            ],
            'Value': [
                self.results_summary['total_samples_original'],
                self.results_summary['total_samples_selected'],
                f"{self.results_summary['sample_retention_rate']:.2f}",
                self.results_summary['total_features_original'],
                self.results_summary['total_features_selected']
            ]
        })
        summary_df.to_csv(f'{output_prefix}_summary.csv', index=False)
        print(f"Сохранено: {output_prefix}_summary.csv")
        
        # Детали по скважинам
        well_details_list = []
        for well, details in self.results_summary['well_details'].items():
            well_details_list.append({
                'Well': well,
                'Original': details['original'],
                'Selected': details['selected'],
                'Retention_Rate': f"{details['retention_rate']:.2f}",
                'Features': details['selected_features']
            })
        well_details_df = pd.DataFrame(well_details_list)
        well_details_df.to_csv(f'{output_prefix}_well_details.csv', index=False)
        print(f"Сохранено: {output_prefix}_well_details.csv")
        
        return self


def main():
    """Основная функция"""
    print("="*60)
    print("АНАЛИЗ И ОТБОР ПРОБ УГЛЕВОДОРОДОВ")
    print("="*60)
    
    # Параметры
    MIN_SAMPLES_RATIO = 0.72  # Минимум 72% проб (чтобы гарантированно > 70%)
    MIN_FEATURES = 10        # Минимум 10 log-отношений
    
    # Инициализация
    selector = HydrocarbonSampleSelector(
        min_samples_ratio=MIN_SAMPLES_RATIO,
        min_features=MIN_FEATURES
    )
    
    # Загрузка и обработка данных
    selector.load_data('Composite_data.txt')
    selector.preprocess_data()
    selector.clr_transform()
    
    # Отбор проб и признаков
    selector.fit()
    
    # Визуализация
    selector.visualize_similarity(save_path_prefix='results')
    selector.visualize_variance(save_path_prefix='results')
    
    # Сохранение результатов
    selector.save_results(output_prefix='results')
    
    print("\n" + "="*60)
    print("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО")
    print("="*60)
    
    return selector


if __name__ == "__main__":
    result = main()
