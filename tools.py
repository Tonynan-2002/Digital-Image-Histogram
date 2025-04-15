from ctypes import sizeof
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Tuple, List, Optional, Union
import gc

class HistogramEqualizationPipeline:
    """
    直方图均衡化管道
    
    该类实现了直方图均衡化的基本操作，包括加载图像、计算直方图、绘制直方图、显示图像、直方图均衡化、颜色直方图均衡化等。
    """
    def __init__(self, color_level: int = 256):
        self.image = None
        self.image_equalized = None
        self.histogram = None
        self.histogram_equalized = None
        self.colorspace = None
        self.bins = color_level

        print("HistogramEqualizationPipeline initialized successfully.")
    
    def load_image(self, image_path: str, grayscale: bool = False) -> None:
        """
        Load an image file
        
        Args:
            image_path: Path to the image file
            grayscale: Whether to load in grayscale mode
        
        The function will load the image file and store it in the "image" attribute.
        If using grayscale mode, the image will be converted to grayscale. Otherwise, it will automatically judge the color space of the image.
        """

        if grayscale:
            self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            self.colorspace = 'gray'
            if self.image is None:
                raise FileNotFoundError(f"Something goes wrrong while loading the image from: {image_path}")
        else:
            self.image = cv2.imread(image_path)
            if self.image is None:
                raise FileNotFoundError(f"Something goes wrrong while loading the image from: {image_path}")
            else:
                if len(self.image.shape) == 3:
                    self.colorspace = 'rgb'
                    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                elif len(self.image.shape) == 2:
                    self.colorspace = 'gray'
                else:
                    raise ValueError("Unsupported image format.")
        
        self.histogram = None
        self.histogram_equalized = None
        self.image_equalized = None
        print("Image loaded successfully.")
        
        gc.collect()
    
    def compute_histogram(self) -> np.ndarray:
        """
        Compute the histogram of an image

        Args:
            image: Input image
            bins: Number of histogram bins
            range_values: Range of pixel values

        Returns:
            Histogram array
        """
        range_values = (0, self.bins)

        if len(self.image.shape) > 2:  # 彩色图像
            hist = np.zeros((3, self.bins))
            for i in range(3):
                hist[i], _ = np.histogram(self.image[:, :, i], bins=self.bins, range=range_values)
        else:  # 灰度图像
            hist, _ = np.histogram(self.image, bins=self.bins, range=range_values)

        self.histogram = hist
        return hist
    
    def plot_histogram(self) -> None:
        """
        Compute and plot the histogram of the loaded image
        
        Args:
            bins: Number of histogram bins
            range_values: Range of pixel values
        
        The function will compute the histogram of the loaded image and plot it using the "plot_histogram" function.
        """
        range_values = (0, self.bins)

        if self.histogram is None:
            self.compute_histogram()

        plot_histogram(self.histogram, title=f"Histogram of {self.colorspace} image")
    
    def display_image_with_histogram(self, title: str = "Image with Histogram") -> None:
        """
        Display the loaded image with its histogram
        
        Args:
            title: Title of the displayed image
        
        The function will display the loaded image with its histogram using the "display_image" function.
        """


        # 创建新的图形窗口
        plt.figure(figsize=(12, 5))
        
        # 左侧显示图像
        plt.subplot(121)
        if len(self.image.shape) == 2:  # 灰度图像
            plt.imshow(self.image, cmap='gray')
        else:  # 彩色图像
            plt.imshow(self.image)
        plt.title(f"{self.colorspace.upper()} Image")
        plt.axis('off')
        
        # 右侧显示直方图
        plt.subplot(122)
        if len(self.histogram.shape) > 1:  # 彩色图像直方图
            colors = ['r', 'g', 'b']
            for i, c in enumerate(colors):
                plt.plot(self.histogram[i], color=c)
            plt.legend(['Red', 'Green', 'Blue'])
        else:  # 灰度图像直方图
            plt.plot(self.histogram, color='black')
        plt.title(f"Histogram of {self.colorspace} image")
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 设置总标题并显示
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def histogram_equalization(self, method: str = 'global') -> None:

        print("Using {} method for histogram equalization...".format(method))
        if method == 'global':
            if self.colorspace == 'gray':
                self.image_equalized = cv2.equalizeHist(self.image.astype(np.uint8))
                self.histogram_equalized = self.compute_histogram(self.image_equalized, bins=256, range_values=(0, 256))
                print("Global histogram equalization (gray level) completed.")
                
            elif self.colorspace == 'rgb':
                self.color_histogram_equalization()

        elif method == 'local':
            """still in construction"""
            pass
        else:
            raise ValueError("Invalid method. Please choose 'global' or 'local'.")

        print("Histogram equalization completed.")

    def display_original_vs_equalized(self, title: str = "Original vs Equalized Image") -> None:
        """
        Display the original image and the equalized image side by side

        Args:
            title: Title of the displayed image
        The function will display the original image and the equalized image side by side using the "display_image" function.
        """

        if self.image_equalized is None:
            self.histogram_equalization()
        # 创建新的图形窗口
        if self.colorspace == 'gray':
            plt.figure(figsize=(12, 5))
            
            # 左侧显示原始图像
            plt.subplot(121)
            plt.imshow(self.image, cmap='gray')
            plt.title(f"{self.colorspace.upper()} Image")
            plt.axis('off')
            
            # 右侧显示均衡化后的图像
            plt.subplot(122)
            plt.imshow(self.image_equalized, cmap='gray')
            plt.title(f"{self.colorspace.upper()} Image Equalized")
            plt.axis('off')
            
        elif self.colorspace == 'rgb':
            plt.figure(figsize=(12, 10))
            
            # 显示原始RGB图像
            plt.subplot(221)
            plt.imshow(self.image)
            plt.title(f"{self.colorspace.upper()} Image")
            plt.axis('off')
            
            # 显示HSV均衡化结果
            plt.subplot(222)
            plt.imshow(self.image_equalized[0])
            plt.title(f"{self.colorspace.upper()} Image Equalized with HSV")
            plt.axis('off')
            
            # 显示YCrCb均衡化结果
            plt.subplot(223)
            plt.imshow(self.image_equalized[1])
            plt.title(f"{self.colorspace.upper()} Image Equalized with YCrCb")
            plt.axis('off')
            
            # 显示RGB均衡化结果
            plt.subplot(224)
            plt.imshow(self.image_equalized[2])
            plt.title(f"{self.colorspace.upper()} Image Equalized with RGB")
            plt.axis('off')
            
        else:
            raise ValueError("Invalid colorspace. Please choose 'gray' or 'rgb'.")
        
        # 设置总标题并显示
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def histogram_comparison(self) -> None:
        """
        比较原始图像和均衡化后的图像的直方图

        参数:
            original_image: 原始图像
            equalized_image: 均衡化后的图像
        """
        if self.image is None:
            raise ValueError("No image loaded. Please load an image first.")

        if self.histogram is None:
            self.compute_histogram()
        
        if self.image_equalized is None:
            self.histogram_equalization()
        
        if self.colorspace == 'gray':

            # 绘制直方图
            plt.figure(figsize=(12, 5))

            plt.subplot(121)
            plt.plot(self.histogram, color='black')
            plt.title('Original Image Histogram')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.subplot(122)
            plt.plot(self.histogram_equalized, color='black')
            plt.title('Equalized Image Histogram')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.suptitle('Histogram Comparison')
            plt.tight_layout()
            plt.show()

        elif self.colorspace == 'rgb':

            plt.figure(figsize=(12, 10))
            colors = ['r', 'g', 'b']
            
            # 显示原始RGB图像
            plt.subplot(221)
            for i, c in enumerate(colors):
                plt.plot(self.histogram[i], color=c)
            plt.legend(['Red', 'Green', 'Blue'])
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.title(f"{self.colorspace.upper()} Image")

            
            # 显示HSV均衡化结果
            plt.subplot(222)
            for i, c in enumerate(colors):
                plt.plot(self.histogram_equalized[0, i], color=c)
            plt.legend(['Red', 'Green', 'Blue'])
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.title(f"{self.colorspace.upper()} Image Equalized with HSV")

            
            # 显示YCrCb均衡化结果
            plt.subplot(223)
            for i, c in enumerate(colors):
                plt.plot(self.histogram_equalized[1, i], color=c)
            plt.legend(['Red', 'Green', 'Blue'])
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.title(f"{self.colorspace.upper()} Image Equalized with YCrCb")

            
            # 显示RGB均衡化结果
            plt.subplot(224)
            for i, c in enumerate(colors):
                plt.plot(self.histogram_equalized[2, i], color=c)
            plt.legend(['Red', 'Green', 'Blue'])
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.title(f"{self.colorspace.upper()} Image Equalized with RGB")

            plt.suptitle("Histogram comparison of different equalization method")
            plt.tight_layout()
            plt.show()

    
    def color_histogram_equalization(self) -> np.ndarray:
        """
        对彩色图像进行直方图均衡化
        
        参数:
            image: 输入彩色图像 (RGB格式)
            method: 均衡化方法，可选 'hsv', 'ycrcb', 或 'rgb'
            
        返回:
            均衡化后的彩色图像
        """

        if len(self.image.shape) != 3:
            raise ValueError("输入必须是彩色图像")
        
        # 初始化四维数组存储所有方法的结果
        # 维度: [方法数, 高度, 宽度, 通道数]
        self.image_equalized = np.zeros((3, *self.image.shape), dtype=self.image.dtype)
        self.histogram_equalized = np.zeros((3, 3, self.bins))
        
        # HSV方法 (索引0)
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        self.image_equalized[0] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        range_values = (0, self.bins)
        for i in range(3):
            self.histogram_equalized[0, i], _ = np.histogram(self.image_equalized[0, :, :, i], bins=self.bins, range=range_values)
        
        
        # YCrCb方法 (索引1)
        ycrcb = cv2.cvtColor(self.image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        self.image_equalized[1] = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        for i in range(3):
            self.histogram_equalized[1, i], _ = np.histogram(self.image_equalized[1, :, :, i], bins=self.bins, range=range_values)
        
        # RGB方法 (索引2)
        for i in range(3):
            self.image_equalized[2, :, :, i] = cv2.equalizeHist(self.image[:, :, i])
        for i in range(3):
            self.histogram_equalized[2, i], _ = np.histogram(self.image_equalized[2, :, :, i], bins=self.bins, range=range_values)
            
        return self.image_equalized

        
    

def load_image(image_path: str, grayscale: bool = False) -> np.ndarray:
    """
    ## Load image file
    
    ### Args:
        image_path: Path to the image file
        grayscale: Whether to load in grayscale mode
        
    ### Returns:
        Loaded image array
    """
    if grayscale:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
    
    if img is None:
        raise FileNotFoundError(f"Something goes wrrong while loading the image from: {image_path}")
    
    return img


def compute_histogram(image: np.ndarray, bins: int = 256, range_values: Tuple[int, int] = (0, 256)) -> np.ndarray:
    """
    Calculate the histogram of an image
    
    Args:
        image: Input image
        bins: Number of histogram bins
        range_values: Range of pixel values
        
    Returns:
        Histogram array
    """
    if len(image.shape) > 2:  # 彩色图像
        hist = np.zeros((3, bins))
        for i in range(3):
            hist[i], _ = np.histogram(image[:, :, i], bins=bins, range=range_values)
        return hist
    else:  # 灰度图像
        hist, _ = np.histogram(image, bins=bins, range=range_values)
        return hist


def plot_histogram(histogram: np.ndarray, title: str = "Histogram", color: Union[str, List[str]] = ['r', 'g', 'b']) -> None:
    """
    绘制直方图
    
    参数:
        histogram: 直方图数据
        title: 图表标题
        color: 直方图颜色
    """
    plt.figure(figsize=(10, 6))
    
    if len(histogram.shape) > 1:  # 彩色图像直方图
        for i, c in enumerate(color):
            plt.plot(histogram[i], color=c)
        plt.legend(['Red', 'Green', 'Blue'])
    else:  # 灰度图像直方图
        plt.plot(histogram, color='black')
    
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, histogram.shape[-1]])
    plt.grid(True, alpha=0.3)
    plt.show()


def display_image(image: np.ndarray, title: str = "Image") -> None:
    """
    显示图像
    
    参数:
        image: 要显示的图像
        title: 图像标题
    """
    plt.figure(figsize=(8, 8))
    if len(image.shape) == 2:  # 灰度图像
        plt.imshow(image, cmap='gray')
    else:  # 彩色图像
        plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    对灰度图像进行直方图均衡化
    
    参数:
        image: 输入灰度图像
        
    返回:
        均衡化后的图像
    """
    if len(image.shape) > 2:
        raise ValueError("此函数仅适用于灰度图像，请先将彩色图像转换为灰度图像")
    
    # 使用OpenCV的直方图均衡化函数
    equalized = cv2.equalizeHist(image.astype(np.uint8))
    return equalized


def color_histogram_equalization(image: np.ndarray, method: str = 'hsv') -> np.ndarray:
    """
    对彩色图像进行直方图均衡化
    
    参数:
        image: 输入彩色图像 (RGB格式)
        method: 均衡化方法，可选 'hsv', 'ycrcb', 或 'rgb'
        
    返回:
        均衡化后的彩色图像
    """
    if len(image.shape) != 3:
        raise ValueError("输入必须是彩色图像")
    
    if method == 'hsv':
        # 转换到HSV空间
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # 仅对V通道进行均衡化
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        # 转换回RGB
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
    elif method == 'ycrcb':
        # 转换到YCrCb空间
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        # 仅对Y通道进行均衡化
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        # 转换回RGB
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        
    elif method == 'rgb':
        # 对每个通道分别进行均衡化
        result = np.zeros_like(image)
        for i in range(3):
            result[:, :, i] = cv2.equalizeHist(image[:, :, i])
    
    else:
        raise ValueError("不支持的方法，请选择 'hsv', 'ycrcb', 或 'rgb'")
    
    return result


def adaptive_histogram_equalization(image: np.ndarray, clip_limit: float = 2.0, 
                                   tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    应用自适应直方图均衡化 (CLAHE)
    
    参数:
        image: 输入图像
        clip_limit: 对比度限制参数
        tile_grid_size: 网格大小
        
    返回:
        CLAHE处理后的图像
    """
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    if len(image.shape) == 2:  # 灰度图像
        return clahe.apply(image.astype(np.uint8))
    
    else:  # 彩色图像
        # 转换到LAB颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        # 仅对L通道应用CLAHE
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        # 转换回RGB
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def compute_gradient_histogram(image: np.ndarray, bins: int = 36) -> np.ndarray:
    """
    计算图像梯度方向的直方图
    
    参数:
        image: 输入灰度图像
        bins: 直方图的柱数
        
    返回:
        梯度方向直方图
    """
    # 确保图像是灰度的
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # 计算x和y方向的梯度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度幅值和方向
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x) * 180 / np.pi  # 转换为度
    
    # 将角度转换为0-360度范围
    angle[angle < 0] += 360
    
    # 计算梯度方向的直方图，使用梯度幅值作为权重
    hist, _ = np.histogram(angle, bins=bins, range=(0, 360), weights=magnitude)
    
    return hist


def compare_histograms(hist1: np.ndarray, hist2: np.ndarray, method: str = 'correlation') -> float:
    """
    比较两个直方图的相似度
    
    参数:
        hist1: 第一个直方图
        hist2: 第二个直方图
        method: 比较方法，可选 'correlation', 'chi-square', 'intersection', 'bhattacharyya'
        
    返回:
        相似度得分
    """
    if method == 'correlation':
        return cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)
    elif method == 'chi-square':
        return cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CHISQR)
    elif method == 'intersection':
        return cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_INTERSECT)
    elif method == 'bhattacharyya':
        return cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
    else:
        raise ValueError("不支持的比较方法")