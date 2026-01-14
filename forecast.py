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
from sklearn.model_selection import train_test_split
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
        
    def _detailed_autocorrelation_analysis(self, max_lag=24):
        """Детальный анализ автокорреляции"""
        try:
            from statsmodels.tsa.stattools import acf, pacf
            
            data = self.data['value'].dropna()
            
            autocorr_values = acf(data, nlags=max_lag, fft=True)
            partial_autocorr_values = pacf(data, nlags=max_lag)
            
            analysis = {
                'significant_lags': [],
                'seasonality_12': False,
                'seasonality_6': False,
                'trend_strength': 'слабый',
                'white_noise_test': False,
                'summary': ''
            }
            
            significance_threshold = 1.96 / np.sqrt(len(data))
            
            for lag in range(1, max_lag + 1):
                if abs(autocorr_values[lag]) > significance_threshold:
                    analysis['significant_lags'].append({
                        'lag': lag,
                        'acf': autocorr_values[lag],
                        'pacf': partial_autocorr_values[lag]
                    })
            
            seasonal_lags = [item for item in analysis['significant_lags'] 
                           if item['lag'] % 12 == 0 or item['lag'] == 12]
            if seasonal_lags:
                analysis['seasonality_12'] = True
            
            semi_seasonal_lags = [item for item in analysis['significant_lags'] 
                                if item['lag'] % 6 == 0 or item['lag'] == 6]
            if semi_seasonal_lags:
                analysis['seasonality_6'] = True
            
            if autocorr_values[1] > 0.8:
                analysis['trend_strength'] = 'очень сильный'
            elif autocorr_values[1] > 0.6:
                analysis['trend_strength'] = 'сильный'
            elif autocorr_values[1] > 0.4:
                analysis['trend_strength'] = 'умеренный'
            elif autocorr_values[1] > 0.2:
                analysis['trend_strength'] = 'слабый'
            else:
                analysis['trend_strength'] = 'очень слабый'
        
            if len(analysis['significant_lags']) == 0:
                analysis['white_noise_test'] = True
            
            summary = "ВЫВОДЫ ДЛЯ SARIMA МОДЕЛИ:\n"
            summary += "=" * 50 + "\n"
            
            if analysis['seasonality_12']:
                summary += "✓ Обнаружена СИЛЬНАЯ СЕЗОННОСТЬ 12 месяцев\n"
                summary += "  Рекомендации:\n"
                summary += "  • Использовать s=12 в сезонных параметрах\n"
                summary += "  • Рассмотреть P, D, Q ≠ 0\n"
            elif analysis['seasonality_6']:
                summary += "✓ Обнаружена СЛАБАЯ СЕЗОННОСТЬ 6 месяцев\n"
                summary += "  Рекомендации:\n"
                summary += "  • Использовать s=6 или s=12\n"
                summary += "  • Рассмотреть P=1, D=0, Q=1\n"
            else:
                summary += "✗ Сезонность НЕ обнаружена\n"
                summary += "  Рекомендации:\n"
                summary += "  • Использовать ARIMA вместо SARIMA\n"
                summary += "  • Или установить P=D=Q=0\n"
            
            summary += "=" * 50 + "\n"
            
            summary += "РЕКОМЕНДАЦИИ ПО ПАРАМЕТРАМ (p,d,q):\n"
            
            pacf_significant = [item for item in analysis['significant_lags'] 
                              if abs(item['pacf']) > significance_threshold and item['lag'] <= 3]
            if pacf_significant:
                max_pacf_lag = max([item['lag'] for item in pacf_significant])
                summary += f"• p ≈ {max_pacf_lag} (по PACF)\n"
            else:
                summary += "• p ≈ 0-2 (слабые автокорреляции)\n"
            
            if not analysis['is_stationary']:
                summary += "• d = 1-2 (ряд нестационарный)\n"
            else:
                summary += "• d = 0-1 (ряд стационарный)\n"
            
            acf_significant = [item for item in analysis['significant_lags'] 
                             if abs(item['acf']) > significance_threshold and item['lag'] <= 3]
            if acf_significant:
                max_acf_lag = max([item['lag'] for item in acf_significant])
                summary += f"• q ≈ {max_acf_lag} (по ACF)\n"
            else:
                summary += "• q ≈ 0-1 (слабые частные автокорреляции)\n"
            
            summary += "=" * 50 + "\n"
            summary += "ПРИМЕРЫ КОНФИГУРАЦИЙ:\n"
            summary += "1. Без сезонности: (1,1,1)(0,0,0,0)\n"
            summary += "2. С сезонностью: (1,1,1)(1,1,1,12)\n"
            summary += "3. Слабые корреляции: (0,1,1)(0,1,1,12)\n"
            
            analysis['summary'] = summary
            return analysis
            
        except Exception as e:
            return {'summary': f"Ошибка анализа: {str(e)}"}
        
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
        analysis_results['autocorrelation_analysis'] = self._detailed_autocorrelation_analysis()
        
        analysis_text = f"АНАЛИЗ АВТОКОРРЕЛЯЦИИ:\n"
        analysis_text += f"1. Тест Дики-Фуллера: {'Стационарный' if analysis_results['is_stationary'] else 'Нестационарный'}\n"
        analysis_text += f"2. P-value: {analysis_results['adf_pvalue']:.4f}\n\n"
        analysis_text += analysis_results['autocorrelation_analysis']['summary']
        
        analysis_results['autocorrelation_summary'] = analysis_text
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
        
        self.feature_columns = [col for col in df.columns if col != 'value']
        
        self.X = df[self.feature_columns]
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
            
            if hasattr(model, 'feature_names_in_'):
                pass
            else:
                try:
                    model.feature_names_ = self.feature_columns
                except:
                    pass 
            
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
            
            validation_success, validation_message = self.validate_feature_names()
            if not validation_success:
                print(f"Предупреждение: {validation_message}")
            
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
            if hasattr(self, 'feature_columns'):
                current_data = self.X.iloc[-1:][self.feature_columns].copy()
            else:
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
                current_data_ensure = current_data[self.feature_columns]
                
                current_scaled = self.scaler.transform(current_data_ensure)
                
                pred = self.regression_model.predict(current_scaled)[0]
                predictions.append(pred)
                
                new_row = self._create_next_features(current_data, pred, i + 1, steps)
                
                if hasattr(self, 'feature_columns'):
                    new_row = new_row[self.feature_columns]
                    
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
            
            if hasattr(self, 'feature_columns'):
                new_data = new_data[self.feature_columns]
            
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
            
            if 'day_of_week' in new_data.columns:
                new_data['day_of_week'] = forecast_date.dayofweek
            if 'month' in new_data.columns:
                new_data['month'] = forecast_date.month
            if 'quarter' in new_data.columns:
                new_data['quarter'] = forecast_date.quarter
            if 'year' in new_data.columns:
                new_data['year'] = forecast_date.year
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
        
    def validate_feature_names(self):
        """Проверка соответствия имен признаков"""
        if not hasattr(self, 'feature_columns'):
            return False, "Признаки не определены"
        
        if not hasattr(self, 'regression_model'):
            return False, "Модель не обучена"
        
        try:
            if hasattr(self.regression_model, 'feature_names_in_'):
                model_features = list(self.regression_model.feature_names_in_)
            else:
                model_features = self.feature_columns
            
            if set(model_features) != set(self.feature_columns):
                return False, f"Несоответствие признаков. Модель: {model_features}, Данные: {self.feature_columns}"
            
            return True, "Признаки соответствуют"
            
        except Exception as e:
            return False, f"Ошибка проверки признаков: {str(e)}"
        
        # Добавить новый метод в класс TimeSeriesForecaster
    def create_sliding_window_features(self, window_size=12):
        """Создание признаков для обучения с помощью sliding window"""
        if self.data is None:
            return False, "Данные не загружены"
        
        try:
            df = self.data.copy()
            values = df['value'].values
            
            # Создаем признаки с помощью sliding window
            X, y = [], []
            for i in range(window_size, len(values)):
                X.append(values[i-window_size:i])
                y.append(values[i])
            
            X = np.array(X)
            y = np.array(y)
            
            # Разделяем на train/test
            X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=42,shuffle=False)
            
            self.sliding_window_data = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'window_size': window_size,
                
            }
            
            return True, f"Sliding window данные подготовлены. Окно: {window_size}, Train: {len(X_train)}, Test: {len(X_test)}"
        
        except Exception as e:
            return False, f"Ошибка создания sliding window признаков: {str(e)}"


    def train_sliding_window_xgboost(self, n_estimators=100, max_depth=3, learning_rate=0.1):
        """Обучение XGBoost на данных с sliding window"""
        if not hasattr(self, 'sliding_window_data'):
            return False, "Сначала создайте sliding window данные"
        
        try:
            X_train = self.sliding_window_data['X_train']
            X_test = self.sliding_window_data['X_test']
            y_train = self.sliding_window_data['y_train']
            y_test = self.sliding_window_data['y_test']
            window_size = self.sliding_window_data['window_size']
            
            # Создаем и обучаем XGBoost модель
            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                eval_metric='rmse'
            )
            
            model.fit(X_train, y_train)
            
            # Прогноз на тестовых данных
            y_pred = model.predict(X_test)
            
            # Вычисляем метрики
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Сохраняем модель и результаты
            self.sliding_window_model = model
            self.sliding_window_results = {
                'model': model,
                'window_size': window_size,
                'metrics': {
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R2': r2
                },
                'predictions': y_pred,
                'actual': y_test,
                'params': {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate
                }
            }
            
            return True, f"XGBoost (sliding window) обучен. Окно: {window_size}, RMSE: {rmse:.2f}, R2: {r2:.2f}"
        
        except Exception as e:
            return False, f"Ошибка обучения XGBoost: {str(e)}"


    def plot_sliding_window_results(self):
        """Визуализация результатов sliding window XGBoost"""
        if not hasattr(self, 'sliding_window_results'):
            return False, "Модель sliding window не обучена"
        
        try:
            
            results = self.sliding_window_results
            y_test = results['actual']
            y_pred = results['predictions']
            window_size = results['window_size']
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # График 1: Фактические vs Прогнозные значения
            
            axes[0, 0].plot(y_test, label='Скользящее окно', color='blue', alpha=0.7)
            axes[0, 0].plot(y_pred, label='Прогноз XGBoost', color='red', linestyle='--', alpha=0.7)
            axes[0, 0].set_title(f'XGBoost прогноз (скользящее окно={window_size})')
            axes[0, 0].set_xlabel('Индекс тестовой выборки')
            axes[0, 0].set_ylabel('Значение')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # График 2: Ошибки прогноза
            errors = y_test - y_pred
            axes[0, 1].plot(errors, color='purple', linewidth=1)
            axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[0, 1].fill_between(range(len(errors)), 0, errors, 
                                where=errors>0, color='green', alpha=0.3, label='Завышение прогноза')
            axes[0, 1].fill_between(range(len(errors)), 0, errors, 
                                where=errors<=0, color='red', alpha=0.3, label='Занижение прогноза')
            axes[0, 1].set_title('Ошибки прогноза')
            axes[0, 1].set_xlabel('Индекс тестовой выборки')
            axes[0, 1].set_ylabel('Ошибка')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # График 3: Распределение ошибок
            axes[1, 0].hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
            axes[1, 0].set_title('Распределение ошибок прогноза')
            axes[1, 0].set_xlabel('Ошибка')
            axes[1, 0].set_ylabel('Частота')
            axes[1, 0].grid(True, alpha=0.3)
            
            # График 4: Важность признаков (если доступно)
            if hasattr(results['model'], 'feature_importances_'):
                importances = results['model'].feature_importances_
                feature_indices = range(len(importances))
                
                axes[1, 1].bar(feature_indices, importances, color='teal', alpha=0.7)
                axes[1, 1].set_title(f'Важность признаков (лагов {window_size})')
                axes[1, 1].set_xlabel('Номер лага')
                axes[1, 1].set_ylabel('Важность')
                axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()
            
            return True, image_base64
        
        except Exception as e:
            return False, f"Ошибка визуализации: {str(e)}"


    def compare_all_models(self):
        """Сравнение всех моделей, включая sliding window XGBoost"""
        comparison = {}
        
        if 'sarima' in self.results:
            comparison['SARIMA'] = self.results['sarima']['metrics']
        
        if 'regression' in self.results:
            comparison['Regression'] = self.results['regression']['metrics']
        
        if hasattr(self, 'sliding_window_results'):
            comparison['XGBoost (Sliding Window)'] = self.sliding_window_results['metrics']
        
        if not comparison:
            return False, "Нет обученных моделей для сравнения"
        
        # Определяем лучшую модель по RMSE
        best_model = min(comparison.items(), key=lambda x: x[1]['RMSE'])
        
        return True, {
            'comparison': comparison,
            'best_model': best_model[0],
            'best_rmse': best_model[1]['RMSE']
        }