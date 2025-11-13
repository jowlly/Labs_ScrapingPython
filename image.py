import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
import os

def create_directories():
    """Создание необходимых директорий для сохранения результатов"""
    directories = ['colors_output','denoising_output', 'contours_output', 'image_transformation', 'morphology_output']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def color(image1_p,show=False):
    """Анализ цветовых пространств и сегментации"""
    # Загрузка изображений
    img_title=image1_p.split('/')[-1].split('.')[0]
    image1 = cv2.imread(image1_p)

    if image1 is None:
        print("Ошибка: Не удалось загрузить одно из изображений")
        return

    # Конвертация из BGR в RGB для корректного отображения
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    # Отображение исходных изображений
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(image1_rgb)
    axes[0].set_title('Изображение 1 - Исходное')
    axes[0].axis('off')

    plt.tight_layout()
    if show:
        plt.show()
    plt.close()
    

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
        plt.savefig(f'colors_output/{img_title} color_spaces.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        return hsv_image, lab_image, ycrcb_image

    # Применяем к первому изображению
    hsv1, lab1, ycrcb1 = demonstrate_color_spaces(image1, "Изображение 1")

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
        plt.savefig(f'colors_output/{img_title} enhance.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()

    # Применяем улучшение
    enhance_image_hsv(image1)

    def skin_detection_comparison(image):
        """Сравнение сегментации кожи в разных цветовых пространствах"""
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
        plt.savefig(f'colors_output/{img_title} skin_detection.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()

    # Применяем сегментацию
    skin_detection_comparison(image1)

def detect_and_show_contours(noiseimage_p,show=False):
    """
    Функция для выделения и визуализации контуров
    """
    # Загрузка изображения
    img_title=noiseimage_p.split('/')[-1].split('.')[0].split('.')[0]
    noiseimage = cv2.imread(noiseimage_p)
    if noiseimage is None:
        print(f"Ошибка: Не удалось загрузить изображение {noiseimage_p}")
        return None, None
        
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
    plt.savefig(f'contours_output/{img_title} contours_analysis.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    
    print(f"Найдено контуров: {len(contours)}")
    
    return contours, hierarchy

def calculate_mse(img1, img2):
    """Вычисление среднеквадратичной ошибки"""
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)

def calculate_psnr(img1, img2):
    """Вычисление пикового отношения сигнал-шум"""
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def generate_salt_pepper_noise(img, noise_percent):
    """Генерация шума 'соль-перец'"""
    height, width, channels = img.shape
    mask = np.ones((height, width, channels))
    mask_size = height * width
    
    num_zeros = int(mask_size * noise_percent)
    
    for channel in range(channels):
        indices_to_change = np.random.choice(mask_size, num_zeros, replace=False)
        mask[indices_to_change % height, indices_to_change // height, channel] = 0
    
    return (img * mask).astype(np.uint8), mask

def display_comparison(images, titles, rows, cols, figsize=(15, 10), image_title="comparison",show=False):
    """Функция для отображения сравнения изображений"""
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'denoising_output/{image_title}.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def denoising_analysis(image_path,show=False):
    """Анализ шума и методов очистки изображений"""
    img_title = image_path.split('/')[-1].split('.')[0]
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: Не удалось загрузить изображение '{image_path}'")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print("=== Часть 1: Сжатие JPEG и оценка качества ===")
    
    flag, encoded_img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    print(f"Сжатие успешно: {flag}")
    print(f"Размер до сжатия: {img.size} байт")
    print(f"Размер после сжатия: {encoded_img.size} байт")
    
    decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    decoded_img_rgb = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
    
    mse = calculate_mse(img_rgb, decoded_img_rgb)
    psnr = calculate_psnr(img_rgb, decoded_img_rgb)
    print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
    print(f"Пиковое отношение сигнал-шум (PSNR): {psnr:.2f} dB")
    
    images = [img_rgb, decoded_img_rgb]
    titles = ['Оригинал', f'Сжатое JPEG (PSNR: {psnr:.2f} dB)']
    display_comparison(images, titles, 1, 2, (12, 6), f"{img_title} compare_with_clean",show)
    
    print("\n=== Часть 2: Non-Local Means Denoising ===")
    
    noisy_img_gaussian = random_noise(img_rgb / 255.0, mode="gaussian") * 255.0
    noisy_img_gaussian = np.clip(noisy_img_gaussian, 0, 255).astype(np.uint8)
    noisy_img_gaussian_bgr = cv2.cvtColor(noisy_img_gaussian, cv2.COLOR_RGB2BGR)
    
    denoised_nlm = cv2.fastNlMeansDenoisingColored(
        noisy_img_gaussian_bgr, None, h=10, hColor=10, 
        templateWindowSize=7, searchWindowSize=21
    )
    denoised_nlm_rgb = cv2.cvtColor(denoised_nlm, cv2.COLOR_BGR2RGB)
    
    psnr_noisy = calculate_psnr(img_rgb, noisy_img_gaussian)
    psnr_denoised = calculate_psnr(img_rgb, denoised_nlm_rgb)
    print(f"PSNR между оригиналом и зашумленным изображением: {psnr_noisy:.2f} dB")
    print(f"PSNR между оригиналом и очищенным изображением: {psnr_denoised:.2f} dB")
    
    images = [noisy_img_gaussian, denoised_nlm_rgb]
    titles = [
        f'Гауссовский шум (PSNR: {psnr_noisy:.2f} dB)', 
        f'Non-Local Means (PSNR: {psnr_denoised:.2f} dB)'
    ]
    display_comparison(images, titles, 1, 2, (12, 6), f"{img_title} compressing",show)
    
    print("\n=== Часть 3: Обработка шума 'соль-перец' ===")
    
    noisy_img_salt_pepper, _ = generate_salt_pepper_noise(img_rgb, 0.1)
    noisy_img_salt_pepper_bgr = cv2.cvtColor(noisy_img_salt_pepper, cv2.COLOR_RGB2BGR)
    
    denoised_nlm_sp = cv2.fastNlMeansDenoisingColored(
        noisy_img_salt_pepper_bgr, None, h=10, hColor=10,
        templateWindowSize=7, searchWindowSize=21
    )
    denoised_nlm_sp_rgb = cv2.cvtColor(denoised_nlm_sp, cv2.COLOR_BGR2RGB)
    
    denoised_median = cv2.medianBlur(noisy_img_salt_pepper_bgr, 5)
    denoised_median_rgb = cv2.cvtColor(denoised_median, cv2.COLOR_BGR2RGB)
    
    psnr_noisy_sp = calculate_psnr(img_rgb, noisy_img_salt_pepper)
    psnr_nlm_sp = calculate_psnr(img_rgb, denoised_nlm_sp_rgb)
    psnr_median = calculate_psnr(img_rgb, denoised_median_rgb)
    
    print(f"PSNR зашумленное 'соль-перец': {psnr_noisy_sp:.2f} dB")
    print(f"PSNR Non-Local Means: {psnr_nlm_sp:.2f} dB")
    print(f"PSNR Медианный фильтр: {psnr_median:.2f} dB")
    
    images = [
        noisy_img_salt_pepper, 
        denoised_nlm_sp_rgb, 
        denoised_median_rgb
    ]
    titles = [
        f'Шум "соль-перец" (PSNR: {psnr_noisy_sp:.2f} dB)',
        f'Non-Local Means (PSNR: {psnr_nlm_sp:.2f} dB)',
        f'Медианный фильтр (PSNR: {psnr_median:.2f} dB)'
    ]
    display_comparison(images, titles, 1, 3, (15, 5), f"{img_title} noise",show)
    
    print("\n=== Сравнение методов денойзинга ===")
    
    methods = ["Гауссовский шум", "Non-Local Means", "Шум 'соль-перец'", "NLM для 'соль-перец'", "Медианный фильтр"]
    psnr_values = [psnr_noisy, psnr_denoised, psnr_noisy_sp, psnr_nlm_sp, psnr_median]
    
    print("\nСравнение методов по PSNR:")
    for method, psnr_val in zip(methods, psnr_values):
        print(f"{method}: {psnr_val:.2f} dB")
    
    all_images = [
        img_rgb, noisy_img_gaussian, denoised_nlm_rgb,
        noisy_img_salt_pepper, denoised_nlm_sp_rgb, denoised_median_rgb
    ]
    all_titles = [
        'Оригинал',
        f'Гауссовский шум\n({psnr_noisy:.1f} dB)',
        f'Non-Local Means\n({psnr_denoised:.1f} dB)',
        f'Шум "соль-перец"\n({psnr_noisy_sp:.1f} dB)',
        f'NLM для "соль-перец"\n({psnr_nlm_sp:.1f} dB)',
        f'Медианный фильтр\n({psnr_median:.1f} dB)'
    ]
    
    plt.figure(figsize=(18, 12))
    for i, (image, title) in enumerate(zip(all_images, all_titles)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(image)
        plt.title(title, fontsize=12)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'denoising_output/{img_title} compare_with_original.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    
def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    """Функция для изменения яркости и контраста"""
    adjusted = image.astype(float) * contrast + brightness
    adjusted = np.clip(adjusted, 0, 255)
    return adjusted.astype(np.uint8)

def geometric_transformations(image_path,show=False):
    """Геометрические преобразования и фильтрация изображений"""
    print("=== ГЕОМЕТРИЧЕСКИЕ ПРЕОБРАЗОВАНИЯ ИЗОБРАЖЕНИЙ ===")

    img_title=image_path.split('/')[-1].split('.')[0]
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: Не удалось загрузить изображение '{image_path}'")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"Original shape: {img.shape}")
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title('Исходное изображение')
    plt.axis('off')
    plt.savefig(f'image_transformation/{img_title} original.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

    def display_images(images, titles, rows, cols, figsize=(15, 10), image_title="multiple"):
        """Функция для отображения нескольких изображений"""
        plt.figure(figsize=figsize)
        for i, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(rows, cols, i+1)
            if len(img.shape) == 2:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'image_transformation/{image_title}.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    print("\n=== 1. МАСШТАБИРОВАНИЕ ===")

    res_up_cubic = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    res_up_linear = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    res_down_area = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    res_down_cubic = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)

    images_scaling = [img, res_up_cubic, res_down_area, res_down_cubic]
    titles_scaling = ['Original', 'Увеличение 2x (INTER_CUBIC)', 
                     'Уменьшение 0.3x (INTER_AREA)', 'Уменьшение 0.3x (INTER_CUBIC)']
    display_images(images_scaling, titles_scaling, 2, 2, image_title=f"{img_title} scale")

    print("\n=== 2. СДВИГ ===")
    rows, cols, _ = img.shape

    M_translation = np.float32([[1, 0, 100], [0, 1, 50]])
    dst_translation = cv2.warpAffine(img, M_translation, (cols, rows))

    display_images([img, dst_translation], ['Original', 'Сдвиг (100,50)'], 1, 2, image_title=f"{img_title} translation")

    print("\n=== 3. ПОВОРОТ ===")

    alpha = 45
    scale = 1
    center = ((cols-1)/2.0, (rows-1)/2.0)
    M_rotation = cv2.getRotationMatrix2D(center, alpha, scale)
    dst_rotation = cv2.warpAffine(img, M_rotation, (cols, rows))

    alpha2 = -30
    M_rotation2 = cv2.getRotationMatrix2D(center, alpha2, scale)
    dst_rotation2 = cv2.warpAffine(img, M_rotation2, (cols, rows))

    images_rotation = [img, dst_rotation, dst_rotation2]
    titles_rotation = ['Original', 'Поворот +45°', 'Поворот -30°']
    display_images(images_rotation, titles_rotation, 1, 3, image_title=f"{img_title} rotate")

    print("\n=== 4. АФФИННОЕ ПРЕОБРАЗОВАНИЕ ===")

    pts1 = np.float32([[455, 180], [450, 100], [230, 180]])
    pts2 = np.float32([[500, 100], [400, 50], [100, 300]])

    img_with_points = img.copy()
    for point in pts1:
        cv2.circle(img_with_points, tuple(point.astype(int)), 5, (0, 0, 255), -1)

    M_affine = cv2.getAffineTransform(pts1, pts2)
    dst_affine = cv2.warpAffine(img, M_affine, (cols, rows))

    for point in pts2:
        cv2.circle(dst_affine, tuple(point.astype(int)), 5, (0, 0, 255), -1)

    display_images([img_with_points, dst_affine], 
                   ['Исходное с точками', 'Аффинное преобразование'], 1, 2, image_title=f"{img_title} affine")

    print("\n=== 5. МАСШТАБИРОВАНИЕ ЧЕРЕЗ АФФИННОЕ ПРЕОБРАЗОВАНИЕ ===")
    alpha_scale = 2
    beta_scale = 2
    pts1_scale = np.float32([[455, 180], [450, 100], [230, 180]])
    pts2_scale = np.float32([[455*alpha_scale, 180*beta_scale], 
                            [450*alpha_scale, 100*beta_scale], 
                            [230*alpha_scale, 180*beta_scale]])

    M_affine_scale = cv2.getAffineTransform(pts1_scale, pts2_scale)
    dst_affine_scale = cv2.warpAffine(img, M_affine_scale, 
                                     (cols*alpha_scale, rows*beta_scale), 
                                     cv2.INTER_LINEAR)

    display_images([img, dst_affine_scale], 
                   ['Original', f'Аффинное масштабирование ({alpha_scale}x)'], 1, 2, image_title=f"{img_title} affine_scale")

    print("\n=== 6. ПЕРСПЕКТИВНОЕ ПРЕОБРАЗОВАНИЕ ===")

    img_test2 = cv2.imread(image_path)
    if img_test2 is None:
        print(f"Предупреждение: Не удалось загрузить {image_path}. Пропускаем перспективное преобразование.")
    else:
        rows_s, cols_s, ch = img_test2.shape
        pts1_perspective = np.float32([[56, 65], [368, 52], [28, 287], [389, 390]])
        pts2_perspective = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

        M_perspective = cv2.getPerspectiveTransform(pts1_perspective, pts2_perspective)
        dst_perspective = cv2.warpPerspective(img_test2, M_perspective, (300, 300))

        img_test2_points = img_test2.copy()
        for point in pts1_perspective:
            cv2.circle(img_test2_points, tuple(point.astype(int)), 5, (0, 0, 255), -1)

        images_perspective = [img_test2_points, dst_perspective]
        titles_perspective = ['test2 с точками', 'Перспективное преобразование']
        
        display_images(images_perspective, titles_perspective, 1, 2, image_title=f"{img_title} perspective_translation")

    print("\n=== 7. ИЗМЕНЕНИЕ ЯРКОСТИ И КОНТРАСТА ===")

    bright_high = adjust_brightness_contrast(img, brightness=50, contrast=1.0) 
    bright_low = adjust_brightness_contrast(img, brightness=-50, contrast=1.0) 
    contrast_high = adjust_brightness_contrast(img, brightness=0, contrast=1.5) 
    contrast_low = adjust_brightness_contrast(img, brightness=0, contrast=0.5)  
    both_adjusted = adjust_brightness_contrast(img, brightness=30, contrast=1.3)

    images_bc = [img, bright_high, bright_low, contrast_high, contrast_low, both_adjusted]
    titles_bc = ['Original', 'Яркость +50', 'Яркость -50', 
                 'Контраст x1.5', 'Контраст x0.5', 'Ярк. +30, Контр. x1.3']
    display_images(images_bc, titles_bc, 2, 3, image_title=f"{img_title} brightness")

    print("\n=== 8. ФИЛЬТРАЦИЯ ИЗОБРАЖЕНИЙ ===")

    blur_mean = cv2.blur(img, (5, 5))           
    blur_gaussian = cv2.GaussianBlur(img, (5, 5), 0)  
    blur_median = cv2.medianBlur(img, 5)        

    kernel_sharpen = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
    sharpened = cv2.filter2D(img, -1, kernel_sharpen)

    bilateral = cv2.bilateralFilter(img, 9, 75, 75)

    images_filters = [img, blur_mean, blur_gaussian, blur_median, sharpened, bilateral]
    titles_filters = ['Original', 'Усредняющий фильтр', 'Гауссовский фильтр', 
                      'Медианный фильтр', 'Повышение резкости', 'Биллатеральный фильтр']
    display_images(images_filters, titles_filters, 2, 3, image_title=f"{img_title} filters")

    print("\n=== ОБЪЯСНЕНИЕ ПРЕОБРАЗОВАНИЙ ===")
    explanation = """
    ПРЕОБРАЗОВАНИЯ ДЛЯ ПРЕДОБРАБОТКИ ИЗОБРАЖЕНИЙ:

    Геометрические преобразования:
    • Масштабирование - приведение изображений к единому размеру
    • Поворот и сдвиг - аугментация данных, компенсация наклона камеры
    • Аффинные преобразования - коррекция перспективных искажений

    Изменение яркости и контраста:
    • Нормализация освещения между разными снимками
    • Улучшение видимости деталей в темных/светлых областях
    • Подготовка для алгоритмов компьютерного зрения

    Фильтрация:
    • Удаление шума и артефактов
    • Сглаживание для уменьшения влияния шума
    • Повышение резкости для выделения важных деталей
    • Сохранение границ при удалении шума (билинейная фильтрация)

    Эти преобразования помогают улучшить качество входных данных для последующего анализа
    нейронными сетями и классическими алгоритмами компьютерного зрения.
    """
    print(explanation)

    print("Обработка завершена!")

def draw_img(img, title="Image", figsize=(8, 6), show=False):
    """Функция для отображения одного изображения"""
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'morphology_output/{title}.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def draw_multiple_images(images, titles, rows, cols, figsize=(15, 10), image_title="multiple", show=False):
    """Функция для отображения нескольких изображений"""
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i+1)
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'morphology_output/{image_title}.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def morphological_operations(image_path,show=False):
    """Морфологические операции над изображениями"""
    img_title = image_path.split('/')[-1].split('.')[0]
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: Не удалось загрузить изображение '{image_path}'")
        return

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("=== Базовые морфологические операции ===")

    print("\n1. Эрозия с разными структурными элементами:")

    kernel_cross_3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    print("Крестообразное ядро 3x3:\n", kernel_cross_3)

    img_erosion_cross_3 = cv2.morphologyEx(img_gray, cv2.MORPH_ERODE, kernel_cross_3)
    draw_img(img_erosion_cross_3, f"{img_title} Эрозия (CROSS 3x3)",show=show)

    kernel_ellipse_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    print("Эллиптическое ядро 5x5:\n", kernel_ellipse_5)

    img_erosion_ellipse_5 = cv2.morphologyEx(img_gray, cv2.MORPH_ERODE, kernel_ellipse_5)
    draw_img(img_erosion_ellipse_5, f"{img_title} Эрозия (ELLIPSE 5x5)",show=show)

    print("\n2. Дилатация:")
    kernel_cross_3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    img_dilation = cv2.morphologyEx(img_gray, cv2.MORPH_DILATE, kernel_cross_3)
    draw_img(img_dilation, f"{img_title} Дилатация (CROSS 3x3)",show=show)

    print("\n3. Top-hat преобразование:")
    kernel_cross_6 = cv2.getStructuringElement(cv2.MORPH_CROSS, (6,6))
    img_tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel_cross_6)
    draw_img(img_tophat, f"{img_title} Top-hat (CROSS 6x6)",show=show)

    print("\n4. Комплексные морфологические операции:")

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (6,6))

    img_open = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
    img_close = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
    img_tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    img_blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

    images_complex = [img_gray, img_open, img_close, img_tophat, img_blackhat]
    titles_complex = [
        "Исходное изображение", 
        "Открытие (OPEN)", 
        "Закрытие (CLOSE)", 
        "Top-hat", 
        "Black-hat"
    ]
    draw_multiple_images(images_complex, titles_complex, 2, 3, (16, 10), f"{img_title} complex",show=show)

    print("\n=== Операция Hit-or-Miss ===")

    input_image = np.array((
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 255, 255, 255, 0, 0, 0, 0, 255, 0],
        [0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0],
        [0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 255, 0, 0, 0, 255, 255, 0, 0],
        [0, 0, 255, 255, 255, 0, 0, 0, 255, 0, 0],
        [0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), dtype="uint8")

    kernel_1 = np.array((
        [0,   1, 0],
        [-1,  1, 1],
        [-1, -1, 0]), dtype="int")

    kernel_2 = np.array((
        [0, -1, -1],
        [1,  1, -1],
        [0,  1,  0]), dtype="int")

    kernel_3 = np.array((
        [-1, -1, 0],
        [-1,  1, 1],
        [0,   1, 0]), dtype="int")

    kernel_4 = np.array((
        [0,  1,  0],
        [1,  1, -1],
        [0, -1, -1]), dtype="int")

    output_image_1 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel_1)
    output_image_2 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel_2)
    output_image_3 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel_3)
    output_image_4 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel_4)

    output_image = output_image_1 | output_image_2 | output_image_3 | output_image_4

    rate = 50
    input_image_large = cv2.resize(input_image, None, fx=rate, fy=rate, interpolation=cv2.INTER_NEAREST)
    output_image_large = cv2.resize(output_image, None, fx=rate, fy=rate, interpolation=cv2.INTER_NEAREST)

    images_hitmiss = [input_image_large, output_image_large]
    titles_hitmiss = ["Исходное бинарное изображение", "Результат Hit-or-Miss"]
    draw_multiple_images(images_hitmiss, titles_hitmiss, 1, 2, (12, 6), f"{img_title} hit-or-miss",show=show)

    print("\nАнализ результатов:")
    print("1. Эрозия - уменьшает объекты, убирает шум")
    print("2. Дилатация - увеличивает объекты, заполняет пробелы")
    print("3. Открытие - эрозия + дилатация, убирает мелкие объекты")
    print("4. Закрытие - дилатация + эрозия, заполняет мелкие отверстия")
    print("5. Top-hat - выделяет светлые объекты на темном фоне")
    print("6. Black-hat - выделяет темные объекты на светлом фоне")
    print("7. Hit-or-Miss - поиск специфических шаблонов в изображении")

    print("\nОбработка завершена!")

if __name__ == "__main__":
    create_directories()
    
    print("=== АНАЛИЗ ЦВЕТОВЫХ ПРОСТРАНСТВ ===")
    color('image1.jpg')
    color('image2.jpg')
    
    print("\n=== АНАЛИЗ КОНТУРОВ ===")
    contours, hierarchy = detect_and_show_contours('image1.jpg')
    contours, hierarchy = detect_and_show_contours('image2.jpg')
    
    print("\n=== АНАЛИЗ ШУМА И МЕТОДОВ ОЧИСТКИ ===")
    denoising_analysis('image1.jpg')
    denoising_analysis('image2.jpg')
    
    print("\n=== ГЕОМЕТРИЧЕСКИЕ ПРЕОБРАЗОВАНИЯ ===")
    geometric_transformations('image1.jpg')
    geometric_transformations('image2.jpg')
    
    print("\n=== МОРФОЛОГИЧЕСКИЕ ОПЕРАЦИИ ===")
    morphological_operations('image1.jpg')
    morphological_operations('image2.jpg')
    
    print("\n=== ВСЕ АНАЛИЗЫ ЗАВЕРШЕНЫ ===")