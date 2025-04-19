from ctypes import sizeof
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Tuple, List, Optional, Union
import os
import gc

class HistogramEqualizationPipeline:
    """
    Histogram Equalization Pipeline
    
    This class implements basic operations for histogram equalization, including image loading, 
    histogram computation, histogram plotting, image display, histogram equalization, 
    and color histogram equalization.
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

        if len(self.image.shape) > 2:  # RGB image
            hist = np.zeros((3, self.bins))
            for i in range(3):
                hist[i], _ = np.histogram(self.image[:, :, i], bins=self.bins, range=range_values)
        else:  # gray level image
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


        # Create new figure window
        plt.figure(figsize=(12, 5))
        
        # Display image on the left
        plt.subplot(121)
        if len(self.image.shape) == 2:  # Grayscale image
            plt.imshow(self.image, cmap='gray')
        else:  # Color image
            plt.imshow(self.image)
        plt.title(f"{self.colorspace.upper()} Image")
        plt.axis('off')
        
        # Display histogram on the right
        plt.subplot(122)
        if len(self.histogram.shape) > 1:  # Color image histogram
            colors = ['r', 'g', 'b']
            for i, c in enumerate(colors):
                plt.plot(self.histogram[i], color=c)
            plt.legend(['Red', 'Green', 'Blue'])
        else:  # Grayscale histogram
            plt.plot(self.histogram, color='black')
        plt.title(f"Histogram of {self.colorspace} image")
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Set main title and display
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def histogram_equalization(self, method: str = 'global') -> None:

        print("Using {} method for histogram equalization...".format(method))
        if method == 'global':
            if self.colorspace == 'gray':
                self.image_equalized = cv2.equalizeHist(self.image.astype(np.uint8))
                self.histogram_equalized = self.compute_histogram()
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
        # Create new figure window
        if self.colorspace == 'gray':
            plt.figure(figsize=(12, 5))
            
            # Display original image on the left
            plt.subplot(121)
            plt.imshow(self.image, cmap='gray')
            plt.title(f"{self.colorspace.upper()} Image")
            plt.axis('off')
            
            # Display equalized image on the right
            plt.subplot(122)
            plt.imshow(self.image_equalized, cmap='gray')
            plt.title(f"{self.colorspace.upper()} Image Equalized")
            plt.axis('off')
            
        elif self.colorspace == 'rgb':
            plt.figure(figsize=(12, 10))
            
            # Display original RGB image
            plt.subplot(221)
            plt.imshow(self.image)
            plt.title(f"{self.colorspace.upper()} Image")
            plt.axis('off')
            
            # Display HSV equalization result
            plt.subplot(222)
            plt.imshow(self.image_equalized[0])
            plt.title(f"{self.colorspace.upper()} Image Equalized with HSV")
            plt.axis('off')
            
            # Display YCrCb equalization result
            plt.subplot(223)
            plt.imshow(self.image_equalized[1])
            plt.title(f"{self.colorspace.upper()} Image Equalized with YCrCb")
            plt.axis('off')
            
            # Display RGB equalization result
            plt.subplot(224)
            plt.imshow(self.image_equalized[2])
            plt.title(f"{self.colorspace.upper()} Image Equalized with RGB")
            plt.axis('off')
            
        else:
            raise ValueError("Invalid colorspace. Please choose 'gray' or 'rgb'.")
        
        # Set main title and display
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def histogram_comparison(self) -> None:
        """
        Compare histograms between original image and equalized image

        Args:
            original_image: Original image
            equalized_image: Equalized image
        """
        if self.image is None:
            raise ValueError("No image loaded. Please load an image first.")

        if self.histogram is None:
            self.compute_histogram()
        
        if self.image_equalized is None:
            self.histogram_equalization()
        
        if self.colorspace == 'gray':

            # Plot histograms
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
            
            # Display original RGB histogram
            plt.subplot(221)
            for i, c in enumerate(colors):
                plt.plot(self.histogram[i], color=c)
            plt.legend(['Red', 'Green', 'Blue'])
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.title(f"{self.colorspace.upper()} Image")

            
            # Display HSV equalization histogram
            plt.subplot(222)
            for i, c in enumerate(colors):
                plt.plot(self.histogram_equalized[0, i], color=c)
            plt.legend(['Red', 'Green', 'Blue'])
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.title(f"{self.colorspace.upper()} Image Equalized with HSV")

            
            # Display YCrCb equalization histogram
            plt.subplot(223)
            for i, c in enumerate(colors):
                plt.plot(self.histogram_equalized[1, i], color=c)
            plt.legend(['Red', 'Green', 'Blue'])
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.title(f"{self.colorspace.upper()} Image Equalized with YCrCb")

            
            # Display RGB equalization histogram
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
        Perform histogram equalization on color images
        
        Args:
            image: Input color image (RGB format)
            method: Equalization method, can be 'hsv', 'ycrcb', or 'rgb'
            
        Returns:
            Color image after histogram equalization
        """

        if len(self.image.shape) != 3:
            raise ValueError("Input must be a color image")
        
        # Initialize 4D array to store results of all methods
        # Dimensions: [num_methods, height, width, channels]
        self.image_equalized = np.zeros((3, *self.image.shape), dtype=self.image.dtype)
        self.histogram_equalized = np.zeros((3, 3, self.bins))
        
        # HSV method (index 0)
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        self.image_equalized[0] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        range_values = (0, self.bins)
        for i in range(3):
            self.histogram_equalized[0, i], _ = np.histogram(self.image_equalized[0, :, :, i], bins=self.bins, range=range_values)
        
        
        # YCrCb method (index 1)
        ycrcb = cv2.cvtColor(self.image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        self.image_equalized[1] = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        for i in range(3):
            self.histogram_equalized[1, i], _ = np.histogram(self.image_equalized[1, :, :, i], bins=self.bins, range=range_values)
        
        # RGB method (index 2)
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
    if len(image.shape) > 2:  # RGB imgae
        hist = np.zeros((3, bins))
        for i in range(3):
            hist[i], _ = np.histogram(image[:, :, i], bins=bins, range=range_values)
        return hist
    else:  # gray level image
        hist, _ = np.histogram(image, bins=bins, range=range_values)
        return hist


def plot_histogram(histogram: np.ndarray, title: str = "Histogram", color: Union[str, List[str]] = ['r', 'g', 'b']) -> None:
    """
    Plot histogram
    
    Args:
        histogram: Histogram data
        title: Chart title
        color: Histogram color
    """
    plt.figure(figsize=(10, 6))
    
    if len(histogram.shape) > 1:  # Color image histogram
        for i, c in enumerate(color):
            plt.plot(histogram[i], color=c)
        plt.legend(['Red', 'Green', 'Blue'])
    else:  # Grayscale histogram
        plt.plot(histogram, color='black')
    
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, histogram.shape[-1]])
    plt.grid(True, alpha=0.3)
    plt.show()


def display_image(image: np.ndarray, title: str = "Image") -> None:
    """
    Display image
    
    Args:
        image: Image to display
        title: Image title
    """
    plt.figure(figsize=(8, 8))
    if len(image.shape) == 2:  # Grayscale image
        plt.imshow(image, cmap='gray')
    else:  # Color image
        plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Perform histogram equalization on grayscale image
    
    Args:
        image: Input grayscale image
        
    Returns:
        Equalized image
    """
    if len(image.shape) > 2:
        raise ValueError("This function only works with grayscale images. Please convert color images to grayscale first")
    
    # Use OpenCV's histogram equalization function
    equalized = cv2.equalizeHist(image.astype(np.uint8))
    return equalized


def color_histogram_equalization(image: np.ndarray, method: str = 'hsv') -> np.ndarray:
    """
    Perform histogram equalization on color images
    
    Args:
        image: Input color image (RGB format)
        method: Equalization method, can be 'hsv', 'ycrcb', or 'rgb'
        
    Returns:
        Color image after histogram equalization
    """
    if len(image.shape) != 3:
        raise ValueError("Input must be a color image")
    
    if method == 'hsv':
        # Convert to HSV space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Only equalize V channel
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        # Convert back to RGB
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
    elif method == 'ycrcb':
        # Convert to YCrCb space
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        # Only equalize Y channel
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        # Convert back to RGB
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        
    elif method == 'rgb':
        # Equalize each channel separately
        result = np.zeros_like(image)
        for i in range(3):
            result[:, :, i] = cv2.equalizeHist(image[:, :, i])
    
    else:
        raise ValueError("Unsupported method, please choose 'hsv', 'ycrcb', or 'rgb'")
    
    return result


def adaptive_histogram_equalization(image: np.ndarray, clip_limit: float = 2.0, 
                                   tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    
    Args:
        image: Input image
        clip_limit: Contrast limit parameter
        tile_grid_size: Grid size
        
    Returns:
        CLAHE processed image
    """
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    if len(image.shape) == 2:  # Grayscale image
        return clahe.apply(image.astype(np.uint8))
    
    else:  # Color image
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        # Only apply CLAHE to L channel
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        # Convert back to RGB
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def compute_gradient_histogram(image: np.ndarray, bins: int = 36) -> np.ndarray:
    """
    Compute gradient direction histogram of image
    
    Args:
        image: Input grayscale image
        bins: Number of histogram bins
        
    Returns:
        Gradient direction histogram
    """
    # Ensure image is grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Calculate x and y gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude and direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x) * 180 / np.pi  # Convert to degrees
    
    # Convert angles to 0-360 degree range
    angle[angle < 0] += 360
    
    # Calculate gradient direction histogram using magnitude as weights
    hist, _ = np.histogram(angle, bins=bins, range=(0, 360), weights=magnitude)
    
    return hist


def compare_histograms(hist1: np.ndarray, hist2: np.ndarray, method: str = 'correlation') -> float:
    """
    Compare similarity between two histograms
    
    Args:
        hist1: First histogram
        hist2: Second histogram
        method: Comparison method, can be 'correlation', 'chi-square', 'intersection', 'bhattacharyya'
        
    Returns:
        Similarity score
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
        raise ValueError("Unsupported comparison method")



def apply_and_display_equalization(image_path):
    """
    Loads an image, converts it to grayscale if necessary, applies global
    Histogram Equalization (HE) and Contrast Limited Adaptive Histogram
    Equalization (CLAHE), and displays the original grayscale image along
    with the HE and CLAHE results, plus the original histogram, in a 2x2
    matplotlib plot.

    Note: Standard AHE is computationally expensive and prone to noise
    amplification. CLAHE is the widely used, improved adaptive method.

    Args:
        image_path (str): The path to the image file.
    """
    # --- 1. Image Loading and Validation ---
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: '{image_path}'")
        return
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image: '{image_path}'")
        return

    # --- 2. Image Preprocessing ---
    # Convert to grayscale if RGB image
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("Detected color image, converted to grayscale")
    else:
        gray_img = img
        print("Detected grayscale image")

    # --- 3. Apply Different Histogram Equalization Methods ---
    # Global Histogram Equalization (HE)
    he_img = cv2.equalizeHist(gray_img)

    # Adaptive Histogram Equalization (AHE)
    # Implement AHE using local histogram equalization
    height, width = gray_img.shape
    ahe_img = np.zeros_like(gray_img)
    tile_size = (32, 32)  # Define local region size
    
    for i in range(0, height, tile_size[0]):
        for j in range(0, width, tile_size[1]):
            # Get local region
            tile = gray_img[i:min(i+tile_size[0], height), 
                          j:min(j+tile_size[1], width)]
            # Apply histogram equalization to local region
            if tile.size > 0:  # Ensure tile is not empty
                ahe_img[i:min(i+tile_size[0], height), 
                       j:min(j+tile_size[1], width)] = cv2.equalizeHist(tile)

    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_img)

    # --- 4. Display Results ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.ravel()

    # Define images and corresponding titles to display
    images = [gray_img, he_img, ahe_img, clahe_img]
    titles = ['Original Grayscale', 
             'Global Histogram Equalization (HE)',
             'Adaptive Histogram Equalization (AHE)',
             'CLAHE']

    # Display all images
    for idx, (img, title) in enumerate(zip(images, titles)):
        axs[idx].imshow(img, cmap='gray')
        axs[idx].set_title(title)
        axs[idx].axis('off')

    plt.tight_layout()
    plt.show()

    # Free memory
    gc.collect()
