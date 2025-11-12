import json
import numpy as np
import pandas as pd
import flet as ft
from datetime import datetime
import os
from data import create_dataset_annotation, create_reorganized_dataset, get_data_by_date
from analytics import CurrencyAnalytics
from forecast import TimeSeriesForecaster 


class MainWindow:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "Анализ данных о валютах"
        self.page.scroll = "adaptive"
        
        self.source_dataset_path = None
        self.annotation_output_path = None
        self.reorganized_dataset_path = None
        
        self.waiting_for_annotation_path = False
        self.waiting_for_reorganized_path = False
        
        self.forecaster = TimeSeriesForecaster()
        self.analytics = CurrencyAnalytics()
        self.analytics_image = ft.Image()
        self.analytics_result_text = ft.Text()
        
        self.pick_source_folder_dialog = ft.FilePicker(
            on_result=self.pick_source_folder_result
        )
        self.pick_annotation_file_dialog = ft.FilePicker(
            on_result=self.pick_annotation_file_result
        )
        self.pick_reorganized_folder_dialog = ft.FilePicker(
            on_result=self.pick_reorganized_folder_result
        )
        self.pick_analytics_file_dialog = ft.FilePicker(
            on_result=self.pick_analytics_file_result
        )
        self.pick_forecast_file_dialog = ft.FilePicker(
            on_result=self.pick_forecast_file_result
        )
        self.pick_model_file_dialog = ft.FilePicker(
            on_result=self.pick_model_file_result
        )
        
        self.page.overlay.extend([
            self.pick_forecast_file_dialog, 
            self.pick_model_file_dialog
        ])
        
        self._create_forecast_tab()
        
        self.page.overlay.extend([self.pick_source_folder_dialog, 
                                 self.pick_annotation_file_dialog, 
                                 self.pick_reorganized_folder_dialog,
                                 self.pick_analytics_file_dialog])
    
        self.source_folder_path = ft.TextField(label="Путь к папке исходного датасета", read_only=True, expand=True)
        self.annotation_file_path = ft.TextField(label="Файл аннотации исходного датасета", read_only=True, expand=True)
        self.reorganized_folder_path = ft.TextField(label="Папка для реорганизованного датасета", read_only=True, expand=True)
        
        self.reorganization_type = ft.Dropdown(
            label="Тип реорганизации",
            options=[
                ft.dropdown.Option("years", "По годам"),
                ft.dropdown.Option("weeks", "По неделям")
            ],
            value="years"
        )
        
        self.date_input = ft.TextField(label="Дата (гггг-мм-дд)", value="2025-09-01")
        
        self.result_text = ft.Text()
        
        self.select_source_folder_btn = ft.ElevatedButton(
            "Выбрать папку исходного датасета",
            on_click=lambda _: self.pick_source_folder_dialog.get_directory_path()
        )
        
        self.create_annotation_btn = ft.ElevatedButton(
            "Создать аннотацию исходного датасета",
            on_click=self.start_create_annotation
        )
        
        self.create_reorganized_btn = ft.ElevatedButton(
            "Создать реорганизованный датасет и аннотацию",
            on_click=self.start_create_reorganized_dataset
        )
        
        self.get_data_btn = ft.ElevatedButton(
            "Получить данные",
            on_click=self.get_data_by_date
        )
        
        self.analytics_file_path = ft.TextField(label="Файл данных (dataset.csv)", read_only=True, expand=True)
        self.select_analytics_file_btn = ft.ElevatedButton(
            "Выбрать файл данных",
            on_click=lambda _: self.pick_analytics_file_dialog.pick_files(
                allowed_extensions=["csv"],
                file_type=ft.FilePickerFileType.CUSTOM
            )
        )
        
        self.start_date_input = ft.TextField(label="Начальная дата (гггг-мм-дд)", value="2000-01-01")
        self.end_date_input = ft.TextField(label="Конечная дата (гггг-мм-дд)", value="2025-10-05")
        self.threshold_input = ft.TextField(label="Порог отклонения", value="100")
        self.month_input = ft.TextField(label="Месяц для анализа (гггг-мм)", value="2025-01")
        
        self.load_data_btn = ft.ElevatedButton(
            "Загрузить и подготовить данные",
            on_click=self.load_and_prepare_data
        )
        
        self.filter_deviation_btn = ft.ElevatedButton(
            "Фильтровать по отклонению",
            on_click=self.filter_by_deviation
        )
        
        self.filter_date_btn = ft.ElevatedButton(
            "Фильтровать по дате",
            on_click=self.filter_by_date_range
        )
        
        self.plot_month_btn = ft.ElevatedButton(
            "Построить график за месяц",
            on_click=self.plot_month_data
        )
        
        self.plot_full_btn = ft.ElevatedButton(
            "Построить график за весь период",
            on_click=self.plot_full_period
        )
        
        self.show_stats_btn = ft.ElevatedButton(
            "Показать статистику",
            on_click=self.show_statistics
        )
        
        self.show_boxplot_btn = ft.ElevatedButton(
            "Построить Boxplot",
            on_click=self.show_boxplot
        )
        
        main_tab = ft.Tab(
            text="Основные функции",
            content=ft.Container(
                content=ft.Column([
                    ft.Row([self.source_folder_path, self.select_source_folder_btn]),
                    ft.Row([self.annotation_file_path, self.create_annotation_btn]),
                    ft.Row([self.reorganized_folder_path, self.create_reorganized_btn]),
                    ft.Row([self.reorganization_type]),
                    ft.Divider(),
                    ft.Row([self.date_input, self.get_data_btn]),
                    self.result_text
                ]),
                padding=20
            )
        )
        
        analytics_tab = ft.Tab(
            text="Аналитика",
            content=ft.Container(
                content=ft.Column([
                    ft.Row([self.analytics_file_path, self.select_analytics_file_btn]),
                    ft.Row([self.load_data_btn]),
                    ft.Divider(),
                    ft.Row([self.start_date_input, self.end_date_input]),
                    ft.Row([self.threshold_input, self.filter_deviation_btn]),
                    ft.Row([self.filter_date_btn]),
                    ft.Row([self.month_input, self.plot_month_btn]),
                    ft.Row([self.plot_full_btn, self.show_stats_btn, self.show_boxplot_btn]),
                    self.analytics_result_text,
                    self.analytics_image
                ], scroll="adaptive"),
                padding=20
            )
        )
        
        forecast_tab = ft.Tab(
            text="Прогнозирование",
            content=self.forecast_tab_content
        )
        
        tabs = ft.Tabs(
            tabs=[main_tab, analytics_tab, forecast_tab],
            expand=True
        )
        
        self.page.add(tabs)
    
    def pick_source_folder_result(self, e: ft.FilePickerResultEvent):
        if e.path:
            self.source_dataset_path = e.path
            self.source_folder_path.value = e.path
            self.page.update()
    
    def pick_annotation_file_result(self, e: ft.FilePickerResultEvent):
        if e.path:
            self.annotation_output_path = e.path
            self.annotation_file_path.value = e.path
            self.page.update()
            
            if self.waiting_for_annotation_path:
                self.waiting_for_annotation_path = False
                self.create_annotation()
    
    def pick_reorganized_folder_result(self, e: ft.FilePickerResultEvent):
        if e.path:
            self.reorganized_dataset_path = e.path
            self.reorganized_folder_path.value = e.path
            self.page.update()
            
            if self.waiting_for_reorganized_path:
                self.waiting_for_reorganized_path = False
                self.create_reorganized_dataset()
    
    def pick_analytics_file_result(self, e: ft.FilePickerResultEvent):
        if e.files and len(e.files) > 0:
            file_path = e.files[0].path
            self.analytics_file_path.value = file_path
            self.page.update()
    
    def start_create_annotation(self, e):
        if not self.source_dataset_path:
            self.result_text.value = "Сначала выберите папку исходного датасета"
            self.page.update()
            return
        
        self.waiting_for_annotation_path = True
        self.result_text.value = "Выберите файл для сохранения аннотации..."
        self.pick_annotation_file_dialog.save_file(
            allowed_extensions=["csv"],
            file_name="annotation.csv"
        )
        self.page.update()
    
    def create_annotation(self):
        try:
            create_dataset_annotation(self.source_dataset_path, self.annotation_output_path)
            self.result_text.value = f"Аннотация создана: {self.annotation_output_path}"
        except Exception as ex:
            self.result_text.value = f"Ошибка при создании аннотации: {str(ex)}"
        self.page.update()
    
    def start_create_reorganized_dataset(self, e):
        if not self.source_dataset_path:
            self.result_text.value = "Сначала выберите папку исходного датасета"
            self.page.update()
            return
        
        self.waiting_for_reorganized_path = True
        self.result_text.value = "Выберите папку для реорганизованного датасета..."
        self.pick_reorganized_folder_dialog.get_directory_path()
        self.page.update()
    
    def create_reorganized_dataset(self):
        try:
            create_reorganized_dataset(
                self.source_dataset_path, 
                self.reorganized_dataset_path, 
                self.reorganization_type.value
            )
            self.result_text.value = f"Реорганизованный датасет создан: {self.reorganized_dataset_path}"
        except Exception as ex:
            self.result_text.value = f"Ошибка при создании реорганизованного датасета: {str(ex)}"
        self.page.update()
    
    def get_data_by_date(self, e):
        if not self.source_dataset_path:
            self.result_text.value = "Сначала выберите папку исходного датасета"
            self.page.update()
            return
        
        date_str = self.date_input.value
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            self.result_text.value = "Неверный формат даты. Используйте гггг-мм-дд."
            self.page.update()
            return
        
        try:
            source_file = os.path.join(self.source_dataset_path, 'dataset.csv')
            value = get_data_by_date(date, source_path=source_file)
            
            if value is None:
                self.result_text.value = f"Данные для даты {date_str} не найдены."
            else:
                self.result_text.value = f"Данные для даты {date_str}: {value}"
        except Exception as ex:
            self.result_text.value = f"Ошибка при получении данных: {str(ex)}"
        
        self.page.update()
    
    def load_and_prepare_data(self, e):
        if not self.analytics_file_path.value:
            self.analytics_result_text.value = "Сначала выберите файл данных"
            self.page.update()
            return
        
        try:
            success, message = self.analytics.load_data(self.analytics_file_path.value)
            if not success:
                self.analytics_result_text.value = message
                self.page.update()
                return
            
            start_date = self.start_date_input.value
            end_date = self.end_date_input.value
            success, message = self.analytics.prepare_data(start_date, end_date)
            
            self.analytics_result_text.value = message
            
        except Exception as ex:
            self.analytics_result_text.value = f"Ошибка при загрузке данных: {str(ex)}"
        
        self.page.update()
    
    def filter_by_deviation(self, e):
        try:
            threshold = float(self.threshold_input.value)
            success, message = self.analytics.filter_data_by_deviation(threshold)
            self.analytics_result_text.value = message
        except Exception as ex:
            self.analytics_result_text.value = f"Ошибка при фильтрации: {str(ex)}"
        
        self.page.update()
    
    def filter_by_date_range(self, e):
        try:
            start_date = self.start_date_input.value
            end_date = self.end_date_input.value
            success, message = self.analytics.filter_data_by_date_range(start_date, end_date)
            self.analytics_result_text.value = message
        except Exception as ex:
            self.analytics_result_text.value = f"Ошибка при фильтрации по дате: {str(ex)}"
        
        self.page.update()
    
    def plot_month_data(self, e):
        try:
            month = self.month_input.value
            success, result = self.analytics.plot_month_data(month)
            
            if success:
                self.analytics_image.src_base64 = result
                self.analytics_result_text.value = f"График за {month} построен"
            else:
                self.analytics_result_text.value = result
            
        except Exception as ex:
            self.analytics_result_text.value = f"Ошибка при построении графика: {str(ex)}"
        
        self.page.update()
    
    def plot_full_period(self, e):
        try:
            success, result = self.analytics.plot_full_period()
            
            if success:
                self.analytics_image.src_base64 = result
                self.analytics_result_text.value = "График за весь период построен"
            else:
                self.analytics_result_text.value = result
            
        except Exception as ex:
            self.analytics_result_text.value = f"Ошибка при построении графика: {str(ex)}"
        
        self.page.update()
    
    def show_statistics(self, e):
        try:
            success, result = self.analytics.get_statistics()
            self.analytics_result_text.value = result
        except Exception as ex:
            self.analytics_result_text.value = f"Ошибка при вычислении статистики: {str(ex)}"
        
        self.page.update()
    
    def show_boxplot(self, e):
        try:
            success, result = self.analytics.show_boxplot()
            
            if success:
                self.analytics_image.src_base64 = result
                self.analytics_result_text.value = "Boxplot построен"
            else:
                self.analytics_result_text.value = result
            
        except Exception as ex:
            self.analytics_result_text.value = f"Ошибка при построении boxplot: {str(ex)}"
        
        self.page.update()
        
    def _create_forecast_tab(self):
        """Создание вкладки прогнозирования"""
        self.forecast_file_path = ft.TextField(
            label="Файл данных для прогнозирования", 
            read_only=True, 
            expand=True
        )
    
        self.sarima_p = ft.TextField(label="p", value="1")
        self.sarima_d = ft.TextField(label="d", value="1")
        self.sarima_q = ft.TextField(label="q", value="1")
        self.seasonal_p = ft.TextField(label="P", value="1")
        self.seasonal_d = ft.TextField(label="D", value="1")
        self.seasonal_q = ft.TextField(label="Q", value="1")
        self.seasonal_s = ft.TextField(label="s", value="12")
        
        self.regression_model_type = ft.Dropdown(
            label="Регрессионная модель",
            options=[
                ft.dropdown.Option("random_forest", "Random Forest"),
                ft.dropdown.Option("xgboost", "XGBoost"),
                ft.dropdown.Option("catboost", "CatBoost"),
                ft.dropdown.Option("linear", "Linear Regression")
            ],
            value="random_forest"
        )
        
        self.hyperparam_sets = ft.TextField(
            label="Наборы гиперпараметров (JSON)",
            multiline=True,
            value=json.dumps([
                {"n_estimators": 100, "max_depth": 10},
                {"n_estimators": 200, "max_depth": 15},
                {"n_estimators": 150, "max_depth": 20}
            ], indent=2)
        )
        
        self.forecast_result_text = ft.Text()
        self.forecast_image = ft.Image()
        self.model_comparison_text = ft.Text()
        
        self.load_forecast_data_btn = ft.ElevatedButton(
            "Загрузить данные для прогнозирования",
            on_click=lambda _: self.pick_forecast_file_dialog.pick_files(
                allowed_extensions=["csv"],
                file_type=ft.FilePickerFileType.CUSTOM
            )
        )
        
        self.analyze_ts_btn = ft.ElevatedButton(
            "Анализ временного ряда",
            on_click=self.analyze_time_series
        )
        
        self.prepare_data_btn = ft.ElevatedButton(
            "Подготовить данные",
            on_click=self.prepare_forecast_data
        )
        
        self.train_sarima_btn = ft.ElevatedButton(
            "Обучить SARIMA",
            on_click=self.train_sarima_model
        )
        
        self.train_regression_btn = ft.ElevatedButton(
            "Обучить регрессию",
            on_click=self.train_regression_model
        )
        
        self.tune_hyperparams_btn = ft.ElevatedButton(
            "Настроить гиперпараметры",
            on_click=self.tune_hyperparameters
        )
        
        self.compare_models_btn = ft.ElevatedButton(
            "Сравнить модели",
            on_click=self.compare_models
        )
        
        self.save_model_btn = ft.ElevatedButton(
            "Сохранить лучшую модель",
            on_click=self.save_best_model
        )
        
        self.load_model_btn = ft.ElevatedButton(
            "Загрузить модель",
            on_click=lambda _: self.pick_model_file_dialog.pick_files(
                allowed_extensions=["pkl", "joblib"],
                file_type=ft.FilePickerFileType.CUSTOM
            )
        )
        
        self.predict_btn = ft.ElevatedButton(
            "Сделать прогноз",
            on_click=self.make_prediction
        )
        
        self.forecast_tab_content = ft.Container(
            content=ft.Column([
                ft.Row([self.forecast_file_path, self.load_forecast_data_btn]),
                
                ft.Divider(),
                
                ft.Row([
                    self.analyze_ts_btn,
                    self.prepare_data_btn
                ]),
                
                ft.Divider(),
                
                ft.Text("SARIMA Параметры:", weight=ft.FontWeight.BOLD),
                ft.Row([
                    self.sarima_p, self.sarima_d, self.sarima_q,
                    self.seasonal_p, self.seasonal_d, self.seasonal_q, self.seasonal_s
                ]),
                
                ft.Row([
                    self.train_sarima_btn,
                    self.regression_model_type,
                    self.train_regression_btn
                ]),
                
                ft.Divider(),
                
                ft.Text("Настройка гиперпараметров:", weight=ft.FontWeight.BOLD),
                self.hyperparam_sets,
                self.tune_hyperparams_btn,
                
                ft.Divider(),
                
                ft.Row([
                    self.compare_models_btn,
                    self.save_model_btn,
                    self.load_model_btn,
                    self.predict_btn
                ]),
                
                self.forecast_result_text,
                self.model_comparison_text,
                self.forecast_image
                
            ], scroll="adaptive"),
            padding=20
        )
    
    def pick_forecast_file_result(self, e: ft.FilePickerResultEvent):
        if e.files and len(e.files) > 0:
            file_path = e.files[0].path
            self.forecast_file_path.value = file_path
            self.page.update()
    
    def pick_model_file_result(self, e: ft.FilePickerResultEvent):
        if e.files and len(e.files) > 0:
            file_path = e.files[0].path
            success, message = self.forecaster.load_model(file_path)
            self.forecast_result_text.value = message
            self.page.update()
    
    def analyze_time_series(self, e):
        if not self.forecast_file_path.value:
            self.forecast_result_text.value = "Сначала выберите файл данных"
            self.page.update()
            return
        
        success, message = self.forecaster.load_data(self.forecast_file_path.value)
        if not success:
            self.forecast_result_text.value = message
            self.page.update()
            return
        
        success, result = self.forecaster.analyze_time_series()
        if success:
            self.forecast_image.src_base64 = result['plot']
            self.forecast_result_text.value = (
                f"Анализ завершен. Стационарность: {result['is_stationary']}\n"
                f"ADF p-value: {result['adf_pvalue']:.4f}"
            )
        else:
            self.forecast_result_text.value = result
        
        self.page.update()
    
    def prepare_forecast_data(self, e):
        success, message = self.forecaster.prepare_data()
        self.forecast_result_text.value = message
        self.page.update()
    
    def train_sarima_model(self, e):
        order = (
            int(self.sarima_p.value),
            int(self.sarima_d.value),
            int(self.sarima_q.value)
        )
        seasonal_order = (
            int(self.seasonal_p.value),
            int(self.seasonal_d.value),
            int(self.seasonal_q.value),
            int(self.seasonal_s.value)
        )
        
        success, message = self.forecaster.train_sarima(order, seasonal_order)
        self.forecast_result_text.value = message
        
        if success:
            # Логируем результаты
            self._log_results("SARIMA", self.forecaster.results['sarima'])
        
        self.page.update()
    
    def train_regression_model(self, e):
        success, message = self.forecaster.train_regression(
            self.regression_model_type.value
        )
        self.forecast_result_text.value = message
        
        if success:
            self._log_results("Regression", self.forecaster.results['regression'])
        
        self.page.update()
    
    def tune_hyperparameters(self, e):
        try:
            param_sets = json.loads(self.hyperparam_sets.value)
            model_type = self.regression_model_type.value
            
            results, best_params, best_score = self.forecaster.hyperparameter_tuning(
                model_type, param_sets
            )
            
            log_message = "Результаты настройки гиперпараметров:\n"
            for result in results:
                log_message += f"Итерация {result['iteration']}: RMSE={result['RMSE']:.4f}, Params={result['params']}\n"
            
            log_message += f"\nЛучшие параметры: {best_params}, RMSE: {best_score:.4f}"
            
            self.forecast_result_text.value = log_message
            self._log_hyperparameter_tuning(results, best_params, best_score)
            
        except Exception as ex:
            self.forecast_result_text.value = f"Ошибка настройки: {str(ex)}"
        
        self.page.update()
    
    def compare_models(self, e):
        success, result = self.forecaster.compare_models()
        if success:
            comparison_text = "Сравнение моделей:\n\n"
            comparison_text += "SARIMA:\n"
            for metric, value in result['comparison']['SARIMA'].items():
                comparison_text += f"  {metric}: {value:.4f}\n"
            
            comparison_text += "\nRegression:\n"
            for metric, value in result['comparison']['Regression'].items():
                comparison_text += f"  {metric}: {value:.4f}\n"
            
            comparison_text += f"\nЛучшая модель: {result['best_model']}"
            
            self.model_comparison_text.value = comparison_text
            
            success, plot = self.forecaster.plot_predictions()
            if success:
                self.forecast_image.src_base64 = plot
            
        else:
            self.model_comparison_text.value = result
        
        self.page.update()
    
    def save_best_model(self, e):
        try:
            import tempfile
            temp_dir = tempfile.gettempdir()
            model_path = os.path.join(temp_dir, "best_model.pkl")
            
            success, message = self.forecaster.save_best_model(model_path)
            self.forecast_result_text.value = message
        except Exception as ex:
            self.forecast_result_text.value = f"Ошибка сохранения: {str(ex)}"
        
        self.page.update()
    
    def make_prediction(self, e):
        """Улучшенный метод прогнозирования в GUI"""
        try:
            if 'sarima' in self.forecaster.results and 'regression' in self.forecaster.results:
                sarima_rmse = self.forecaster.results['sarima']['metrics']['RMSE']
                regression_rmse = self.forecaster.results['regression']['metrics']['RMSE']
                model_type = 'sarima' if sarima_rmse < regression_rmse else 'regression'
            elif hasattr(self.forecaster, 'sarima_fit'):
                model_type = 'sarima'
            elif hasattr(self.forecaster, 'regression_model'):
                model_type = 'regression'
            else:
                self.forecast_result_text.value = "Нет обученных моделей"
                self.page.update()
                return

            success, forecast_results = self.forecaster.predict_new_data(model_type, steps=30)
            
            if success:
                plot_success, plot_image = self.forecaster.plot_forecast(
                    forecast_results, 
                    self.forecaster.data.tail(100)
                )
                
                if plot_success:
                    self.forecast_image.src_base64 = plot_image
            
                report = self._create_forecast_report(forecast_results, model_type)
                self.forecast_result_text.value = report
                
                self._save_forecast_to_file(forecast_results, model_type)
                
            else:
                self.forecast_result_text.value = forecast_results
            
            self.page.update()
            
        except Exception as ex:
            self.forecast_result_text.value = f"Ошибка прогнозирования: {str(ex)}"
            self.page.update()

    def _create_forecast_report(self, forecast_results, model_type):
        """Создание отчета о прогнозе"""
        predictions = forecast_results['predictions']
        
        report = f"Прогноз на 30 шагов (модель: {model_type})\n\n"
        report += f"Первые 10 значений:\n"
        for i, pred in enumerate(predictions[:10]):
            report += f"  День {i+1}: {pred:.2f}\n"
        
        report += f"\nСтатистика прогноза:\n"
        report += f"  Среднее: {np.mean(predictions):.2f}\n"
        report += f"  Стандартное отклонение: {np.std(predictions):.2f}\n"
        report += f"  Минимум: {np.min(predictions):.2f}\n"
        report += f"  Максимум: {np.max(predictions):.2f}\n"
        report += f"  Тренд: {'↗️ Растущий' if predictions[-1] > predictions[0] else '↘️ Падающий'}\n"
        
        if 'confidence_intervals' in forecast_results:
            conf_int = forecast_results['confidence_intervals']
            report += f"\nДоверительные интервалы (95%):\n"
            report += f"  Первый день: [{conf_int.iloc[0, 0]:.2f}, {conf_int.iloc[0, 1]:.2f}]\n"
            report += f"  Последний день: [{conf_int.iloc[-1, 0]:.2f}, {conf_int.iloc[-1, 1]:.2f}]\n"
        
        return report

    def _save_forecast_to_file(self, forecast_results, model_type):
        """Сохранение прогноза в файл"""
        try:
            import tempfile
            import json
            
            log_dir = "forecasts"
            os.makedirs(log_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"forecast_{model_type}_{timestamp}.json"
            filepath = os.path.join(log_dir, filename)
            
            save_data = {
                'timestamp': timestamp,
                'model_type': model_type,
                'predictions': forecast_results['predictions'].tolist() if hasattr(forecast_results['predictions'], 'tolist') else forecast_results['predictions'],
                'metadata': {
                    'data_points': len(forecast_results['predictions']),
                    'mean_prediction': float(np.mean(forecast_results['predictions'])),
                    'std_prediction': float(np.std(forecast_results['predictions']))
                }
            }
            
            if 'confidence_intervals' in forecast_results:
                if hasattr(forecast_results['confidence_intervals'], 'to_dict'):
                    save_data['confidence_intervals'] = forecast_results['confidence_intervals'].to_dict()
                else:
                    save_data['confidence_intervals'] = forecast_results['confidence_intervals']
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            csv_filename = f"forecast_{model_type}_{timestamp}.csv"
            csv_filepath = os.path.join(log_dir, csv_filename)
            
            if 'forecast_df' in forecast_results:
                forecast_results['forecast_df'].to_csv(csv_filepath, index=False)
            else:
                last_date = self.forecaster.data.index[-1]
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1), 
                    periods=len(forecast_results['predictions']), 
                    freq='D'
                )
                
                forecast_df = pd.DataFrame({
                    'date': forecast_dates,
                    'prediction': forecast_results['predictions']
                })
                
                if 'confidence_intervals' in forecast_results:
                    forecast_df['lower_bound'] = forecast_results['confidence_intervals'].iloc[:, 0]
                    forecast_df['upper_bound'] = forecast_results['confidence_intervals'].iloc[:, 1]
                
                forecast_df.to_csv(csv_filepath, index=False)
            
            print(f"Прогноз сохранен: {filepath}")
            
        except Exception as e:
            print(f"Ошибка сохранения прогноза: {str(e)}")
    
    def _log_results(self, model_name, results):
        """Логирование результатов обучения"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, "training_results.txt")
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n=== {model_name} - {datetime.now()} ===\n")
            f.write(f"Параметры: {results.get('order', results.get('type', 'N/A'))}\n")
            f.write("Метрики:\n")
            for metric, value in results['metrics'].items():
                f.write(f"  {metric}: {value}\n")
            f.write("=" * 50 + "\n")
    
    def _log_hyperparameter_tuning(self, results, best_params, best_score):
        """Логирование результатов настройки гиперпараметров"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, "hyperparameter_tuning.txt")
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n=== Настройка гиперпараметров - {datetime.now()} ===\n")
            for result in results:
                f.write(f"Итерация {result['iteration']}: RMSE={result['RMSE']:.4f}\n")
                f.write(f"Параметры: {result['params']}\n")
            f.write(f"Лучшие параметры: {best_params}\n")
            f.write(f"Лучший RMSE: {best_score:.4f}\n")
            f.write("=" * 50 + "\n")


def main(page: ft.Page):
    MainWindow(page)


if __name__ == "__main__":
    ft.app(target=main)