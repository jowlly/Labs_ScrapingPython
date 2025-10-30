import cv2
import numpy as np
import matplotlib.pyplot as plt

def color(image1_p, image2_p):
    # Загрузка изображений
    image1 = cv2.imread(image1_p)
    image2 = cv2.imread(image2_p)

    # Конвертация из BGR в RGB для корректного отображения
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Отображение исходных изображений
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(image1_rgb)
    axes[0].set_title('Изображение 1 - Исходное')
    axes[0].axis('off')

    axes[1].imshow(image2_rgb)
    axes[1].set_title('Изображение 2 - Исходное')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    def demonstrate_color_spaces(image, title):
        """Демонстрация преобразований между различными цветовыми пространствами"""
        
        # Исходное изображение в RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Преобразование в HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_to_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        
        # Преобразование в LAB
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_to_rgb = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
        
        # Преобразование в YCrCb
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb_to_rgb = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2RGB)
        
        # Отображение результатов
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Цветовые пространства - {title}', fontsize=16)
        
        # Первая строка - исходные и преобразованные изображения
        axes[0, 0].imshow(rgb_image)
        axes[0, 0].set_title('Исходное RGB')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(hsv_to_rgb)
        axes[0, 1].set_title('HSV → RGB')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(lab_to_rgb)
        axes[0, 2].set_title('LAB → RGB')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(ycrcb_to_rgb)
        axes[0, 3].set_title('YCrCb → RGB')
        axes[0, 3].axis('off')
        
        # Вторая строка - каналы цветовых пространств
        # HSV каналы
        axes[1, 0].imshow(hsv_image[:, :, 0], cmap='hsv')
        axes[1, 0].set_title('HSV - Hue')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(hsv_image[:, :, 1], cmap='gray')
        axes[1, 1].set_title('HSV - Saturation')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(hsv_image[:, :, 2], cmap='gray')
        axes[1, 2].set_title('HSV - Value')
        axes[1, 2].axis('off')
        
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return hsv_image, lab_image, ycrcb_image

    # Применяем к первому изображению
    hsv1, lab1, ycrcb1 = demonstrate_color_spaces(image1, "Изображение 1")

    # Применяем ко второму изображению
    hsv2, lab2, ycrcb2 = demonstrate_color_spaces(image2, "Изображение 2")

    def enhance_image_hsv(image):
        """Улучшение изображения с использованием HSV пространства"""
        
        # Конвертируем в HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Увеличиваем насыщенность
        hsv_enhanced = hsv.copy()
        hsv_enhanced[:, :, 1] = cv2.multiply(hsv_enhanced[:, :, 1], 1.2)
        hsv_enhanced[:, :, 1] = np.clip(hsv_enhanced[:, :, 1], 0, 255)
        
        # Конвертируем обратно в RGB
        enhanced_rgb = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Сравнение
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].imshow(original_rgb)
        axes[0].set_title('Исходное изображение')
        axes[0].axis('off')
        
        axes[1].imshow(enhanced_rgb)
        axes[1].set_title('Улучшенная насыщенность (HSV)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()

    # Применяем улучшение
    enhance_image_hsv(image1)
    enhance_image_hsv(image2)
    def skin_detection_comparison(image):
        """Сравнение сегментации кожи в разных цветовых пространствах"""
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 480))

        # HSV сегментация кожи
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_skin_hsv = np.array([0, 20, 70])
        upper_skin_hsv = np.array([20, 150, 255])
        mask_hsv = cv2.inRange(hsv, lower_skin_hsv, upper_skin_hsv)
        
        # YCrCb сегментация кожи
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        lower_skin_ycrcb = np.array([0, 135, 85])
        upper_skin_ycrcb = np.array([255, 180, 135])
        mask_ycrcb = cv2.inRange(ycrcb, lower_skin_ycrcb, upper_skin_ycrcb)
        
        # Применяем маски
        result_hsv = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_hsv)
        result_ycrcb = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_ycrcb)
        
        
        # Отображение
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(rgb_image)
        axes[0, 0].set_title('Исходное')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask_hsv, cmap='gray')
        axes[0, 1].set_title('Маска кожи (HSV)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(result_hsv)
        axes[0, 2].set_title('Результат (HSV)')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(rgb_image)
        axes[1, 0].set_title('Исходное')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(mask_ycrcb, cmap='gray')
        axes[1, 1].set_title('Маска кожи (YCrCb)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(result_ycrcb)
        axes[1, 2].set_title('Результат (YCrCb)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()

    # Применяем сегментацию
    skin_detection_comparison(image1)
    skin_detection_comparison(image2)

def detect_and_show_contours(noiseimage_p):
    """
    Функция для выделения и визуализации контуров
    """
    # Загрузка изображения
    noiseimage = cv2.imread(noiseimage_p)
    rgb_image = cv2.cvtColor(noiseimage, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(noiseimage, cv2.COLOR_BGR2GRAY)
    
    # 1. ПРЕДОБРАБОТКА - улучшение для выделения контуров
    # Размытие для уменьшения шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Эквализация гистограммы для улучшения контраста
    enhanced = cv2.equalizeHist(blurred)
    
    # 2. ДЕТЕКЦИЯ КРАЕВ - метод Canny
    edges = cv2.Canny(enhanced, 50, 150)
    
    # 3. ПОИСК КОНТУРОВ
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Исходное изображение
    axes[0,0].imshow(rgb_image)
    axes[0,0].set_title('Исходное изображение')
    axes[0,0].axis('off')
    
    # Серое изображение
    axes[0,1].imshow(gray, cmap='gray')
    axes[0,1].set_title('Серое изображение')
    axes[0,1].axis('off')
    
    # Улучшенное изображение
    axes[0,2].imshow(enhanced, cmap='gray')
    axes[0,2].set_title('После улучшения')
    axes[0,2].axis('off')
    
    # Края Canny
    axes[1,0].imshow(edges, cmap='gray')
    axes[1,0].set_title('Края (Canny)')
    axes[1,0].axis('off')
    
    # Контуры на изображении
    contour_image = rgb_image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    axes[1,1].imshow(contour_image)
    axes[1,1].set_title(f'Найдено контуров: {len(contours)}')
    axes[1,1].axis('off')
    
    # Только контуры
    contours_only = np.zeros_like(rgb_image)
    cv2.drawContours(contours_only, contours, -1, (0, 255, 0), 2)
    axes[1,2].imshow(contours_only)
    axes[1,2].set_title('Только контуры')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Найдено контуров: {len(contours)}")
    
    return contours, hierarchy


if __name__ == "__main__":
    #color('image1.jpg', 'image2.jpg')
    contours, hierarchy = detect_and_show_contours('image2.jpg')
