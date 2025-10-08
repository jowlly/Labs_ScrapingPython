import flet as ft
from datetime import datetime
import os
from data import create_dataset_annotation, create_reorganized_dataset, get_data_by_date
from analytics import CurrencyAnalytics


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
        
        tabs = ft.Tabs(
            tabs=[main_tab, analytics_tab],
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


def main(page: ft.Page):
    MainWindow(page)


if __name__ == "__main__":
    ft.app(target=main)