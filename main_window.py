import flet as ft
from datetime import datetime
import os
from data import create_dataset_annotation, create_reorganized_dataset, get_data_by_date


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
        
        self.pick_source_folder_dialog = ft.FilePicker(
            on_result=self.pick_source_folder_result
        )
        self.pick_annotation_file_dialog = ft.FilePicker(
            on_result=self.pick_annotation_file_result
        )
        self.pick_reorganized_folder_dialog = ft.FilePicker(
            on_result=self.pick_reorganized_folder_result
        )
        
        self.page.overlay.extend([self.pick_source_folder_dialog, 
                                 self.pick_annotation_file_dialog, 
                                 self.pick_reorganized_folder_dialog])
    
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
        
        self.page.add(
            ft.Row([self.source_folder_path, self.select_source_folder_btn]),
            ft.Row([self.annotation_file_path, self.create_annotation_btn]),
            ft.Row([self.reorganized_folder_path, self.create_reorganized_btn]),
            ft.Row([self.reorganization_type]),
            ft.Divider(),
            ft.Row([self.date_input, self.get_data_btn]),
            self.result_text
        )
    
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
    
    def start_create_annotation(self, e):
        """Начинает процесс создания аннотации - запрашивает путь"""
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
        """Создает аннотацию (вызывается после выбора пути)"""
        try:
            create_dataset_annotation(self.source_dataset_path, self.annotation_output_path)
            self.result_text.value = f"Аннотация создана: {self.annotation_output_path}"
        except Exception as ex:
            self.result_text.value = f"Ошибка при создании аннотации: {str(ex)}"
        self.page.update()
    
    def start_create_reorganized_dataset(self, e):
        """Начинает процесс создания реорганизованного датасета - запрашивает путь"""
        if not self.source_dataset_path:
            self.result_text.value = "Сначала выберите папку исходного датасета"
            self.page.update()
            return
        
        self.waiting_for_reorganized_path = True
        self.result_text.value = "Выберите папку для реорганизованного датасета..."
        self.pick_reorganized_folder_dialog.get_directory_path()
        self.page.update()
    
    def create_reorganized_dataset(self):
        """Создает реорганизованный датасет (вызывается после выбора пути)"""
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


def main(page: ft.Page):
    MainWindow(page)


if __name__ == "__main__":
    ft.app(target=main)