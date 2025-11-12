import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

class TimeSeriesForecaster:
    def __init__(self):
        self.data = None
        self.train_data = None
        self.test_data = None
        self.sarima_model = None
        self.regression_model = None
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_data(self, file_path):
        """Загрузка и подготовка данных"""
        try:
            self.data = pd.read_csv(file_path)
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.set_index('date')
            self.data = self.data.sort_index()
            self.data = self.data.dropna()
            return True, "Данные успешно загружены"
        except Exception as e:
            return False, f"Ошибка загрузки данных: {str(e)}"
    
    def analyze_time_series(self):
        """Анализ временного ряда"""
        if self.data is None:
            return False, "Данные не загружены"
        
        analysis_results = {}
        
        result = adfuller(self.data['value'].dropna())
        analysis_results['adf_statistic'] = result[0]
        analysis_results['adf_pvalue'] = result[1]
        analysis_results['is_stationary'] = result[1] < 0.05
        
        analysis_results['autocorrelation'] = {
            'acf': acf(self.data['value'], nlags=20),
            'pacf': pacf(self.data['value'], nlags=20)
        }
        
        analysis_results['basic_stats'] = {
            'mean': self.data['value'].mean(),
            'std': self.data['value'].std(),
            'min': self.data['value'].min(),
            'max': self.data['value'].max()
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(self.data.index, self.data['value'])
        axes[0, 0].set_title('Исходный временной ряд')
        axes[0, 0].set_xlabel('Дата')
        axes[0, 0].set_ylabel('Значение')
        
        plot_acf(self.data['value'], ax=axes[0, 1], lags=20)
        axes[0, 1].set_title('Автокорреляционная функция (ACF)')
        
        plot_pacf(self.data['value'], ax=axes[1, 0], lags=20)
        axes[1, 0].set_title('Частная автокорреляционная функция (PACF)')
        
        axes[1, 1].hist(self.data['value'], bins=30, alpha=0.7)
        axes[1, 1].set_title('Распределение значений')
        axes[1, 1].set_xlabel('Значение')
        axes[1, 1].set_ylabel('Частота')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        
        analysis_results['plot'] = image_base64
        self.analysis_results = analysis_results
        
        return True, analysis_results
    
    def prepare_data(self, test_size=0.2):
        """Подготовка данных для обучения"""
        if self.data is None:
            return False, "Данные не загружены"
        
        split_index = int(len(self.data) * (1 - test_size))
        self.train_data = self.data.iloc[:split_index]
        self.test_data = self.data.iloc[split_index:]
        
        self._create_regression_features()
        
        return True, f"Данные подготовлены: train={len(self.train_data)}, test={len(self.test_data)}"
    
    def _create_regression_features(self, window_size=10):
        """Создание признаков для регрессионной модели"""
        
        df = self.data.copy()
        
        for i in range(1, window_size + 1):
            df[f'lag_{i}'] = df['value'].shift(i)
        
        df['rolling_mean_7'] = df['value'].rolling(window=7).mean()
        df['rolling_std_7'] = df['value'].rolling(window=7).std()
        df['rolling_min_7'] = df['value'].rolling(window=7).min()
        df['rolling_max_7'] = df['value'].rolling(window=7).max()
        
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        
        df = df.dropna()
        
        self.X = df.drop('value', axis=1)
        self.y = df['value']
        
        split_index = int(len(df) * 0.8)
        self.X_train = self.X.iloc[:split_index]
        self.X_test = self.X.iloc[split_index:]
        self.y_train = self.y.iloc[:split_index]
        self.y_test = self.y.iloc[split_index:]
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
    
    def train_sarima(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """Обучение модели SARIMA"""
        try:
            self.sarima_model = SARIMAX(
                self.train_data['value'],
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.sarima_fit = self.sarima_model.fit(disp=False)
            
            sarima_forecast = self.sarima_fit.forecast(steps=len(self.test_data))
            
            mae = mean_absolute_error(self.test_data['value'], sarima_forecast)
            mse = mean_squared_error(self.test_data['value'], sarima_forecast)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.test_data['value'], sarima_forecast)
            
            self.results['sarima'] = {
                'model': self.sarima_fit,
                'order': order,
                'seasonal_order': seasonal_order,
                'metrics': {
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R2': r2
                },
                'predictions': sarima_forecast
            }
            
            return True, f"SARIMA модель обучена. RMSE: {rmse:.2f}, R2: {r2:.2f}"
            
        except Exception as e:
            return False, f"Ошибка обучения SARIMA: {str(e)}"
    
    def train_regression(self, model_type='random_forest', **kwargs):
        """Обучение регрессионной модели"""
        try:
            if model_type == 'linear':
                model = LinearRegression(**kwargs)
            elif model_type == 'random_forest':
                model = RandomForestRegressor(random_state=42, **kwargs)
            elif model_type == 'xgboost':
                model = XGBRegressor(random_state=42, **kwargs)
            elif model_type == 'catboost':
                model = CatBoostRegressor(random_state=42, verbose=False, **kwargs)
            else:
                return False, f"Неизвестный тип модели: {model_type}"
            
            model.fit(self.X_train_scaled, self.y_train)
            
            y_pred = model.predict(self.X_test_scaled)
            
            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)
            
            self.regression_model = model
            self.results['regression'] = {
                'model': model,
                'type': model_type,
                'metrics': {
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R2': r2
                },
                'predictions': y_pred
            }
            
            return True, f"{model_type} модель обучена. RMSE: {rmse:.2f}, R2: {r2:.2f}"
            
        except Exception as e:
            return False, f"Ошибка обучения регрессионной модели: {str(e)}"
    
    def hyperparameter_tuning(self, model_type, param_sets):
        """Настройка гиперпараметров"""
        best_score = float('inf')
        best_params = None
        tuning_results = []
        
        for i, params in enumerate(param_sets):
            if model_type == 'sarima':
                success, message = self.train_sarima(**params)
                if success:
                    score = self.results['sarima']['metrics']['RMSE']
                    tuning_results.append({
                        'iteration': i + 1,
                        'params': params,
                        'RMSE': score
                    })
                    if score < best_score:
                        best_score = score
                        best_params = params
            else:
                success, message = self.train_regression(model_type, **params)
                if success:
                    score = self.results['regression']['metrics']['RMSE']
                    tuning_results.append({
                        'iteration': i + 1,
                        'params': params,
                        'RMSE': score
                    })
                    if score < best_score:
                        best_score = score
                        best_params = params
        
        return tuning_results, best_params, best_score
    
    def compare_models(self):
        """Сравнение моделей"""
        if 'sarima' not in self.results or 'regression' not in self.results:
            return False, "Не все модели обучены"
        
        comparison = {
            'SARIMA': self.results['sarima']['metrics'],
            'Regression': self.results['regression']['metrics']
        }
        
        sarima_rmse = self.results['sarima']['metrics']['RMSE']
        regression_rmse = self.results['regression']['metrics']['RMSE']
        
        best_model = 'SARIMA' if sarima_rmse < regression_rmse else 'Regression'
        
        return True, {
            'comparison': comparison,
            'best_model': best_model
        }
    
    def plot_predictions(self):
        """Визуализация прогнозов"""
        if 'sarima' not in self.results or 'regression' not in self.results:
            return False, "Не все модели обучены"
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        axes[0].plot(self.train_data.index, self.train_data['value'], label='Train')
        axes[0].plot(self.test_data.index, self.test_data['value'], label='True')
        axes[0].plot(self.test_data.index, self.results['sarima']['predictions'], label='SARIMA Prediction')
        axes[0].set_title('SARIMA Прогноз')
        axes[0].legend()
        
        axes[1].plot(self.y_test.index, self.y_test.values, label='True')
        axes[1].plot(self.y_test.index, self.results['regression']['predictions'], label='Regression Prediction')
        axes[1].set_title('Регрессионный Прогноз')
        axes[1].legend()
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        
        return True, image_base64
    
    def save_best_model(self, filepath):
        """Сохранение лучшей модели"""
        try:
            if not self.results:
                return False, "Нет обученных моделей для сохранения"
            
            sarima_rmse = self.results['sarima']['metrics']['RMSE']
            regression_rmse = self.results['regression']['metrics']['RMSE']
            
            if sarima_rmse < regression_rmse:
                best_model = {
                    'type': 'sarima',
                    'model': self.sarima_fit,
                    'metrics': self.results['sarima']['metrics']
                }
            else:
                best_model = {
                    'type': 'regression',
                    'model': self.regression_model,
                    'scaler': self.scaler,
                    'metrics': self.results['regression']['metrics']
                }
            
            joblib.dump(best_model, filepath)
            return True, f"Лучшая модель сохранена в {filepath}"
            
        except Exception as e:
            return False, f"Ошибка сохранения модели: {str(e)}"
    
    def load_model(self, filepath):
        """Загрузка модели"""
        try:
            loaded_data = joblib.load(filepath)
            if loaded_data['type'] == 'sarima':
                self.sarima_fit = loaded_data['model']
            else:
                self.regression_model = loaded_data['model']
                self.scaler = loaded_data['scaler']
            
            return True, "Модель успешно загружена"
        except Exception as e:
            return False, f"Ошибка загрузки модели: {str(e)}"
    
    def predict_new_data(self, model_type, steps=30):
        """Прогноз на новые данные с улучшенной логикой для регрессионных моделей"""
        try:
            if model_type == 'sarima' and hasattr(self, 'sarima_fit'):
                forecast = self.sarima_fit.forecast(steps=steps)
                confidence_intervals = self.sarima_fit.get_forecast(steps=steps).conf_int()
                
                return True, {
                    'predictions': forecast,
                    'confidence_intervals': confidence_intervals,
                    'model_type': 'sarima'
                }
                
            elif model_type == 'regression' and hasattr(self, 'regression_model'):
                return self._regression_forecast(steps)
                
            else:
                return False, "Модель не обучена или неизвестный тип модели"
                
        except Exception as e:
            return False, f"Ошибка прогнозирования: {str(e)}"

    def _regression_forecast(self, steps):
        """Улучшенный прогноз для регрессионных моделей"""
        try:
            current_data = self.X.iloc[-1:].copy()
            predictions = []
            all_predictions_data = []
            
            last_date = self.data.index[-1]
            date_range = pd.date_range(
                start=last_date + pd.Timedelta(days=1), 
                periods=steps, 
                freq='D'
            )
            
            for i in range(steps):
                current_scaled = self.scaler.transform(current_data)
                pred = self.regression_model.predict(current_scaled)[0]
                predictions.append(pred)
                new_row = self._create_next_features(current_data, pred, i + 1, steps)
                current_data = new_row
                all_predictions_data.append({
                    'step': i + 1,
                    'prediction': pred,
                    'features': current_data.values[0]
                })
            
            forecast_df = pd.DataFrame({
                'date': date_range,
                'prediction': predictions
            })
            
            if hasattr(self.regression_model, 'predict'):
                try:
                    test_predictions = self.results['regression']['predictions']
                    test_errors = self.y_test.values - test_predictions
                    std_error = np.std(test_errors)
                    
                    confidence_intervals = pd.DataFrame({
                        'lower': predictions - 1.96 * std_error,
                        'upper': predictions + 1.96 * std_error
                    })
                except:
                    std_value = np.std(predictions)
                    confidence_intervals = pd.DataFrame({
                        'lower': [p * 0.9 for p in predictions],
                        'upper': [p * 1.1 for p in predictions]
                    })
            else:
                confidence_intervals = pd.DataFrame({
                    'lower': [p * 0.9 for p in predictions],
                    'upper': [p * 1.1 for p in predictions]
                })
            
            return True, {
                'predictions': predictions,
                'confidence_intervals': confidence_intervals,
                'forecast_df': forecast_df,
                'prediction_data': all_predictions_data,
                'model_type': 'regression'
            }
            
        except Exception as e:
            return False, f"Ошибка регрессионного прогноза: {str(e)}"

    def _create_next_features(self, current_data, prediction, step, total_steps):
        """Создание признаков для следующего шага прогноза"""
        try:
            new_data = current_data.copy()
            
            lag_columns = [col for col in new_data.columns if col.startswith('lag_')]
            lag_columns_sorted = sorted(lag_columns, 
                                    key=lambda x: int(x.split('_')[1]), 
                                    reverse=True)
            
            for i in range(len(lag_columns_sorted) - 1):
                current_lag = lag_columns_sorted[i]
                next_lag = lag_columns_sorted[i + 1]
                new_data[current_lag] = new_data[next_lag]
            
            if 'lag_1' in new_data.columns:
                new_data['lag_1'] = prediction
            
            self._update_rolling_stats(new_data, prediction, step)
            
            self._update_time_features(new_data, step, total_steps)
            
            return new_data
            
        except Exception as e:
            print(f"Ошибка создания признаков: {str(e)}")
            return current_data

    def _update_rolling_stats(self, new_data, prediction, step):
        """Обновление скользящих статистик"""
        try:
            recent_values = self.data['value'].tail(6).tolist()
            if len(recent_values) >= 6:
                window_values = recent_values[-6:] + [prediction]
                
                new_data['rolling_mean_7'] = np.mean(window_values)
                new_data['rolling_std_7'] = np.std(window_values)
                new_data['rolling_min_7'] = np.min(window_values)
                new_data['rolling_max_7'] = np.max(window_values)
            else:
                if 'rolling_mean_7' in new_data.columns:
                    current_mean = new_data['rolling_mean_7'].iloc[0]
                    new_data['rolling_mean_7'] = (current_mean + prediction) / 2
                    new_data['rolling_std_7'] = new_data['rolling_std_7'] * 1.05 
                
        except Exception as e:
            print(f"Ошибка обновления статистик: {str(e)}")

    def _update_time_features(self, new_data, step, total_steps):
        """Обновление временных признаков"""
        try:
            last_date = self.data.index[-1]
            forecast_date = last_date + pd.Timedelta(days=step)
            
            new_data['day_of_week'] = forecast_date.dayofweek
            new_data['month'] = forecast_date.month
            new_data['quarter'] = forecast_date.quarter
            new_data['year'] = forecast_date.year
            
            new_data['day_of_year'] = forecast_date.dayofyear
            new_data['week_of_year'] = forecast_date.weekofyear
            new_data['is_weekend'] = 1 if forecast_date.dayofweek >= 5 else 0
            
            new_data['sin_month'] = np.sin(2 * np.pi * forecast_date.month / 12)
            new_data['cos_month'] = np.cos(2 * np.pi * forecast_date.month / 12)
            
        except Exception as e:
            print(f"Ошибка обновления временных признаков: {str(e)}")

    def plot_forecast(self, forecast_results, actual_data=None):
        """Визуализация прогноза с доверительными интервалами"""
        try:
            plt.figure(figsize=(15, 8))
            
            if actual_data is not None:
                plt.plot(actual_data.index, actual_data['value'], 
                        label='Исторические данные', color='blue', alpha=0.7)
            
            if forecast_results['model_type'] == 'sarima':
                predictions = forecast_results['predictions']
                conf_int = forecast_results['confidence_intervals']
                
                last_date = self.data.index[-1]
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1), 
                    periods=len(predictions), 
                    freq='D'
                )
                
                plt.plot(forecast_dates, predictions, 
                        label='Прогноз SARIMA', color='red', linewidth=2)
                
                plt.fill_between(forecast_dates, 
                            conf_int.iloc[:, 0], 
                            conf_int.iloc[:, 1], 
                            color='red', alpha=0.2, label='Доверительный интервал')
                
            else: 
                forecast_df = forecast_results['forecast_df']
                conf_int = forecast_results['confidence_intervals']
                
                plt.plot(forecast_df['date'], forecast_df['prediction'], 
                        label='Прогноз регрессии', color='green', linewidth=2)
                
                plt.fill_between(forecast_df['date'], 
                            conf_int['lower'], 
                            conf_int['upper'], 
                            color='green', alpha=0.2, label='Доверительный интервал')
            
            plt.title('Прогноз временного ряда')
            plt.xlabel('Дата')
            plt.ylabel('Значение')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()
            
            return True, image_base64
            
        except Exception as e:
            return False, f"Ошибка построения графика: {str(e)}"

    def get_forecast_metrics(self, forecast_results, actual_values=None):
        """Оценка качества прогноза если есть фактические значения"""
        try:
            if actual_values is None or len(actual_values) != len(forecast_results['predictions']):
                return "Фактические значения для оценки недоступны"
            
            predictions = forecast_results['predictions']
            actual = actual_values
            
            metrics = {
                'MAE': mean_absolute_error(actual, predictions),
                'MSE': mean_squared_error(actual, predictions),
                'RMSE': np.sqrt(mean_squared_error(actual, predictions)),
                'R2': r2_score(actual, predictions),
                'MAPE': np.mean(np.abs((actual - predictions) / actual)) * 100
            }
            
            metrics_text = "Метрики качества прогноза:\n"
            for metric, value in metrics.items():
                metrics_text += f"{metric}: {value:.4f}\n"
            
            return metrics_text
            
        except Exception as e:
            return f"Ошибка расчета метрик: {str(e)}"