from ctypes import sizeof
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Tuple, List, Optional, Union
import gc

# Helper function (assuming it exists elsewhere or defining it here for completeness)
def plot_histogram(hist: np.ndarray, title: str = "Histogram"):
    """
    Plots a histogram. Handles both grayscale (1D) and color (3D) histograms.
    """
    plt.figure()
    if len(hist.shape) > 1:  # Color histogram (assuming shape is (3, bins))
        colors = ['r', 'g', 'b']
        for i, c in enumerate(colors):
            plt.plot(hist[i], color=c)
        plt.legend(['Red', 'Green', 'Blue'])
    else:  # Grayscale histogram
        plt.plot(hist, color='black')

    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


class HistogramEqualizationPipeline:
    """
    直方图均衡化管道

    该类实现了直方图均衡化的基本操作，包括加载图像、计算直方图、绘制直方图、显示图像、直方图均衡化、颜色直方图均衡化等。
    """
    def __init__(self, color_level: int = 256):
        self.image = None
        self.image_equalized = None # For gray it's a single image, for color it's [hsv_eq, ycrcb_eq, rgb_eq]
        self.histogram = None
        self.histogram_equalized = None # For gray it's a single hist, for color it's [hsv_hist, ycrcb_hist, rgb_hist]
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
                raise FileNotFoundError(f"Something goes wrong while loading the image from: {image_path}")
        else:
            img_bgr = cv2.imread(image_path) # Load as BGR first
            if img_bgr is None:
                raise FileNotFoundError(f"Something goes wrong while loading the image from: {image_path}")
            else:
                if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
                    self.colorspace = 'rgb'
                    self.image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Convert to RGB for consistency
                elif len(img_bgr.shape) == 2: # Image was already grayscale
                    self.colorspace = 'gray'
                    self.image = img_bgr
                elif len(img_bgr.shape) == 3 and img_bgr.shape[2] == 1: # Handle single channel image loaded as 3D
                    self.colorspace = 'gray'
                    self.image = img_bgr[:,:,0]
                else:
                     raise ValueError(f"Unsupported image format or shape: {img_bgr.shape}")


        self.histogram = None
        self.histogram_equalized = None
        self.image_equalized = None
        print(f"Image loaded successfully. Colorspace: {self.colorspace}, Shape: {self.image.shape}")

        gc.collect()

    # Adjusted compute_histogram to handle potentially receiving an image as argument
    def compute_histogram(self, image: Optional[np.ndarray] = None, bins: Optional[int] = None, range_values: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Compute the histogram of an image

        Args:
            image: Input image. If None, uses self.image.
            bins: Number of histogram bins. If None, uses self.bins.
            range_values: Range of pixel values. If None, uses (0, self.bins).

        Returns:
            Histogram array
        """
        img_to_process = image if image is not None else self.image
        num_bins = bins if bins is not None else self.bins
        val_range = range_values if range_values is not None else (0, self.bins)

        if img_to_process is None:
             raise ValueError("No image available to compute histogram.")

        if len(img_to_process.shape) > 2 and img_to_process.shape[2] == 3:  # 彩色图像 (assume 3 channels)
            hist = np.zeros((3, num_bins))
            for i in range(3):
                # Ensure channel data is valid before histogram calculation
                channel_data = img_to_process[:, :, i]
                if channel_data.size > 0: # Check if channel is not empty
                     hist[i], _ = np.histogram(channel_data.flatten(), bins=num_bins, range=val_range)
                else:
                     print(f"Warning: Channel {i} has no data.") # Or handle as an error
        elif len(img_to_process.shape) == 2 or (len(img_to_process.shape) == 3 and img_to_process.shape[2] == 1): # 灰度图像 or single channel
             img_flat = img_to_process.flatten()
             if img_flat.size > 0:
                 hist, _ = np.histogram(img_flat, bins=num_bins, range=val_range)
             else:
                 print("Warning: Grayscale image has no data.")
                 hist = np.zeros((num_bins,)) # Return empty histogram
        else:
            raise ValueError(f"Unsupported image shape for histogram calculation: {img_to_process.shape}")


        # Store histogram if computing for the main image
        if image is None:
            self.histogram = hist
        return hist

    def plot_histogram(self) -> None:
        """
        Compute and plot the histogram of the loaded image

        The function will compute the histogram of the loaded image and plot it using the "plot_histogram" helper function.
        """
        if self.image is None:
            print("Please load an image first.")
            return

        if self.histogram is None:
            self.compute_histogram() # Compute for self.image

        if self.histogram is not None:
             plot_histogram(self.histogram, title=f"Histogram of Original {self.colorspace.upper()} image")
        else:
             print("Could not compute or plot histogram.")

    def display_image_with_histogram(self, title: str = "Image with Histogram") -> None:
        """
        Display the loaded image with its histogram

        Args:
            title: Title of the displayed image

        The function will display the loaded image with its histogram using matplotlib.
        """
        if self.image is None:
            print("Please load an image first.")
            return

        if self.histogram is None:
            self.compute_histogram()

        if self.histogram is None: # Check again if computation failed
             print("Histogram data is missing, cannot display.")
             return

        # 创建新的图形窗口
        plt.figure(figsize=(12, 5))

        # 左侧显示图像
        plt.subplot(121)
        if self.colorspace == 'gray':  # 灰度图像
            plt.imshow(self.image, cmap='gray')
        else:  # 彩色图像 (RGB)
            plt.imshow(self.image)
        plt.title(f"Original {self.colorspace.upper()} Image")
        plt.axis('off')

        # 右侧显示直方图
        plt.subplot(122)
        if len(self.histogram.shape) > 1:  # 彩色图像直方图 (shape is (3, bins))
            colors = ['r', 'g', 'b']
            for i, c in enumerate(colors):
                plt.plot(self.histogram[i], color=c)
            plt.legend(['Red', 'Green', 'Blue'])
            plt.title(f"Histogram of Original {self.colorspace.upper()} image")
        else:  # 灰度图像直方图
            plt.plot(self.histogram, color='black')
            plt.title(f"Histogram of Original {self.colorspace.upper()} image")

        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, self.bins]) # Set x-axis limits

        # 设置总标题并显示
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.show()

    def histogram_equalization(self, method: str = 'global') -> None:
        """
        Performs histogram equalization on the loaded image.

        Args:
            method (str): The equalization method. Currently only 'global' is supported.
                          For RGB images, 'global' applies equalization in different color spaces (HSV, YCrCb, RGB).
        """
        if self.image is None:
            print("Please load an image first.")
            return

        print(f"Using {method} method for histogram equalization...")
        if method == 'global':
            if self.colorspace == 'gray':
                # Ensure image is uint8 for equalizeHist
                if self.image.dtype != np.uint8:
                    # Try to scale if it's float/other int type, otherwise raise error
                    if self.image.min() >= 0 and self.image.max() <= 1.0 and self.image.dtype != np.uint8:
                         print("Warning: Image is not uint8. Scaling to 0-255.")
                         img_uint8 = (self.image * 255).astype(np.uint8)
                    elif self.image.min() >= 0 and self.image.max() <= 255 and self.image.dtype != np.uint8:
                         print("Warning: Image is not uint8. Casting.")
                         img_uint8 = self.image.astype(np.uint8)
                    else:
                         raise TypeError(f"cv2.equalizeHist requires uint8 input. Image type is {self.image.dtype} with range [{self.image.min()}, {self.image.max()}]")
                else:
                     img_uint8 = self.image

                self.image_equalized = cv2.equalizeHist(img_uint8)
                # Compute histogram of the equalized image
                self.histogram_equalized = self.compute_histogram(self.image_equalized, bins=self.bins, range_values=(0, self.bins))
                print("Global histogram equalization (gray level) completed.")

            elif self.colorspace == 'rgb':
                # Call the dedicated color equalization method
                self.color_histogram_equalization()
                print("Global histogram equalization (color methods) completed.")

        elif method == 'local':
            # CLAHE (Contrast Limited Adaptive Histogram Equalization) is a common local method
            print("Applying CLAHE (Local Histogram Equalization)...")
            if self.colorspace == 'gray':
                 # Ensure image is uint8
                 if self.image.dtype != np.uint8:
                     if self.image.min() >= 0 and self.image.max() <= 1.0:
                         img_uint8 = (self.image * 255).astype(np.uint8)
                     elif self.image.min() >= 0 and self.image.max() <= 255:
                         img_uint8 = self.image.astype(np.uint8)
                     else:
                          raise TypeError(f"CLAHE requires uint8 input. Image type is {self.image.dtype}")
                 else:
                     img_uint8 = self.image

                 clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                 self.image_equalized = clahe.apply(img_uint8)
                 self.histogram_equalized = self.compute_histogram(self.image_equalized, bins=self.bins, range_values=(0, self.bins))
                 print("CLAHE (gray level) completed.")

            elif self.colorspace == 'rgb':
                 print("Applying CLAHE to intensity channel (e.g., V in HSV or L in LAB)...")
                 # Example: Apply CLAHE on the V channel of HSV
                 hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
                 # Ensure V channel is uint8
                 v_channel = hsv[:, :, 2]
                 if v_channel.dtype != np.uint8:
                     if v_channel.min() >= 0 and v_channel.max() <= 255:
                         v_channel = v_channel.astype(np.uint8)
                     else:
                          raise TypeError(f"V channel must be uint8 for CLAHE. Type is {v_channel.dtype}")

                 clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                 hsv[:, :, 2] = clahe.apply(v_channel)
                 # Store only the result from this specific local method for now
                 # Note: self.image_equalized and self.histogram_equalized will store *only* this result
                 # unlike the global method for color which stores multiple.
                 self.image_equalized = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                 self.histogram_equalized = self.compute_histogram(self.image_equalized, bins=self.bins, range_values=(0, self.bins))
                 print("CLAHE (on HSV's V channel) completed.")
                 # You could add options for LAB's L channel etc. here.

        else:
            raise ValueError("Invalid method. Please choose 'global' or 'local'.")

        gc.collect() # Clean up memory

    def display_original_vs_equalized(self, title: str = "Original vs Equalized Image") -> None:
        """
        Display the original image and the equalized image(s) side by side

        Args:
            title: Title of the displayed image
        """
        if self.image is None:
            print("Please load an image first.")
            return

        if self.image_equalized is None:
            print("No equalized image found. Performing default global equalization...")
            self.histogram_equalization(method='global') # Default to global if not done

        if self.image_equalized is None: # Check again if equalization failed
            print("Equalization failed, cannot display comparison.")
            return

        # Create new figure window
        if self.colorspace == 'gray':
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Left: Original Image
            axes[0].imshow(self.image, cmap='gray')
            axes[0].set_title(f"Original {self.colorspace.upper()} Image")
            axes[0].axis('off')

            # Right: Equalized Image
            axes[1].imshow(self.image_equalized, cmap='gray')
            # Determine the title based on how equalization might have been done
            # This is a bit tricky as we don't store the exact method used if called externally
            # We assume it's the result stored in self.image_equalized
            eq_method_title = "Equalized" # Generic title
            if isinstance(self.image_equalized, np.ndarray) and len(self.image_equalized.shape) == 2:
                 # Likely global or CLAHE result for grayscale
                 eq_method_title = "Globally Equalized" # Or adjust if you know CLAHE was used

            axes[1].set_title(f"{self.colorspace.upper()} Image {eq_method_title}")
            axes[1].axis('off')

        elif self.colorspace == 'rgb':
             # Check if self.image_equalized holds multiple results (from global color eq)
             # or a single result (e.g., from local color eq)
             if isinstance(self.image_equalized, list) or \
                (isinstance(self.image_equalized, np.ndarray) and self.image_equalized.ndim == 4 and self.image_equalized.shape[0] == 3):
                 # Assume global color equalization results (HSV, YCrCb, RGB)
                 fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                 axes_flat = axes.flatten()

                 # Original RGB Image
                 axes_flat[0].imshow(self.image)
                 axes_flat[0].set_title(f"Original {self.colorspace.upper()} Image")
                 axes_flat[0].axis('off')

                 # HSV Equalized Result
                 axes_flat[1].imshow(self.image_equalized[0])
                 axes_flat[1].set_title(f"{self.colorspace.upper()} Equalized (HSV Method)")
                 axes_flat[1].axis('off')

                 # YCrCb Equalized Result
                 axes_flat[2].imshow(self.image_equalized[1])
                 axes_flat[2].set_title(f"{self.colorspace.upper()} Equalized (YCrCb Method)")
                 axes_flat[2].axis('off')

                 # RGB Equalized Result
                 axes_flat[3].imshow(self.image_equalized[2])
                 axes_flat[3].set_title(f"{self.colorspace.upper()} Equalized (RGB Method)")
                 axes_flat[3].axis('off')
             elif isinstance(self.image_equalized, np.ndarray) and self.image_equalized.ndim == 3:
                 # Assume a single color equalized image (e.g., CLAHE on V channel)
                 fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                 # Left: Original Image
                 axes[0].imshow(self.image)
                 axes[0].set_title(f"Original {self.colorspace.upper()} Image")
                 axes[0].axis('off')

                 # Right: Equalized Image
                 axes[1].imshow(self.image_equalized)
                 # Provide a generic title or indicate if CLAHE was likely used
                 axes[1].set_title(f"{self.colorspace.upper()} Image Equalized (e.g., CLAHE)")
                 axes[1].axis('off')
             else:
                  print("Unsupported format for equalized color image display.")
                  return


        else:
            raise ValueError("Invalid colorspace detected.")

        # Set overall title and display
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
        plt.show()

    def histogram_comparison(self) -> None:
        """
        Compares the histograms of the original image and the equalized image(s).
        """
        if self.image is None:
            raise ValueError("No image loaded. Please load an image first.")

        # Ensure histograms are available
        if self.histogram is None:
            print("Computing original histogram...")
            self.compute_histogram()
            if self.histogram is None:
                 raise RuntimeError("Failed to compute original histogram.")


        if self.image_equalized is None or self.histogram_equalized is None:
            print("No equalized results found. Performing default global equalization...")
            self.histogram_equalization(method='global') # Default to global if not done
            if self.image_equalized is None or self.histogram_equalized is None:
                 raise RuntimeError("Failed to compute equalized results.")


        colors = ['r', 'g', 'b']
        x_range = np.arange(self.bins)

        if self.colorspace == 'gray':
            # Plot histograms side-by-side for grayscale
            plt.figure(figsize=(12, 5))

            # Original Histogram
            plt.subplot(121)
            plt.plot(x_range, self.histogram, color='black')
            plt.title('Original Image Histogram')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.xlim([0, self.bins])

            # Equalized Histogram
            plt.subplot(122)
            # Ensure histogram_equalized is 1D for grayscale
            hist_eq = self.histogram_equalized
            if isinstance(hist_eq, np.ndarray) and hist_eq.ndim > 1:
                 print(f"Warning: Expected 1D histogram for grayscale equalized, got {hist_eq.ndim}D. Plotting first dimension.")
                 hist_eq = hist_eq[0] # Attempt to plot the first one if structure is unexpected

            if isinstance(hist_eq, np.ndarray) and hist_eq.ndim == 1:
                 plt.plot(x_range, hist_eq, color='black')
                 plt.title('Equalized Image Histogram')
                 plt.xlabel('Pixel Value')
                 plt.ylabel('Frequency')
                 plt.grid(True, alpha=0.3)
                 plt.xlim([0, self.bins])
            else:
                 plt.text(0.5, 0.5, 'Error: Invalid histogram data', horizontalalignment='center', verticalalignment='center')
                 plt.title('Equalized Image Histogram')


            plt.suptitle('Grayscale Histogram Comparison', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        elif self.colorspace == 'rgb':
            # Plot histograms in a 2x2 grid for color
            plt.figure(figsize=(12, 10))

            # --- Subplot 1: Original Histogram ---
            plt.subplot(221)
            for i, c in enumerate(colors):
                plt.plot(x_range, self.histogram[i], color=c, label=f'Channel {c.upper()}')
            plt.title('Original RGB Histogram')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim([0, self.bins])

            # Check the structure of histogram_equalized
            # Case 1: Global equalization result (list or 3D array)
            if isinstance(self.histogram_equalized, list) or \
               (isinstance(self.histogram_equalized, np.ndarray) and self.histogram_equalized.ndim == 3 and self.histogram_equalized.shape[0] == 3):

                hist_titles = ['HSV Equalized RGB Histogram', 'YCrCb Equalized RGB Histogram', 'RGB Equalized RGB Histogram']
                for idx, title in enumerate(hist_titles):
                    plt.subplot(2, 2, idx + 2) # Subplots 2, 3, 4
                    current_hist_set = self.histogram_equalized[idx]
                    if current_hist_set.shape == (3, self.bins):
                         for i, c in enumerate(colors):
                             plt.plot(x_range, current_hist_set[i], color=c, label=f'Channel {c.upper()}')
                         plt.title(title)
                         plt.xlabel('Pixel Value')
                         plt.ylabel('Frequency')
                         plt.grid(True, alpha=0.3)
                         plt.legend()
                         plt.xlim([0, self.bins])
                    else:
                          plt.text(0.5, 0.5, f'Error: Invalid hist data\nShape: {current_hist_set.shape}', horizontalalignment='center', verticalalignment='center')
                          plt.title(title)

            # Case 2: Single equalization result (e.g., CLAHE) (2D array, shape (3, bins))
            elif isinstance(self.histogram_equalized, np.ndarray) and self.histogram_equalized.ndim == 2 and self.histogram_equalized.shape[0] == 3:
                 plt.subplot(222) # Use the second subplot for the single result
                 for i, c in enumerate(colors):
                     plt.plot(x_range, self.histogram_equalized[i], color=c, label=f'Channel {c.upper()}')
                 plt.title('Equalized RGB Histogram (e.g., CLAHE)')
                 plt.xlabel('Pixel Value')
                 plt.ylabel('Frequency')
                 plt.grid(True, alpha=0.3)
                 plt.legend()
                 plt.xlim([0, self.bins])
                 # Leave other subplots blank or add text
                 plt.subplot(223)
                 plt.text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center')
                 plt.axis('off')
                 plt.subplot(224)
                 plt.text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center')
                 plt.axis('off')

            else:
                 # Handle unexpected histogram_equalized structure
                 plt.subplot(222)
                 plt.text(0.5, 0.5, 'Error: Invalid equalized\nhistogram data structure', horizontalalignment='center', verticalalignment='center')
                 plt.axis('off')
                 plt.subplot(223)
                 plt.text(0.5, 0.5, 'Error', horizontalalignment='center', verticalalignment='center')
                 plt.axis('off')
                 plt.subplot(224)
                 plt.text(0.5, 0.5, 'Error', horizontalalignment='center', verticalalignment='center')
                 plt.axis('off')


            plt.suptitle('Color Histogram Comparison', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        else:
            print(f"Unsupported colorspace '{self.colorspace}' for histogram comparison.")


    # --- color_histogram_equalization ---
    # (Made minor adjustments for consistency and added histogram computation within)
    def color_histogram_equalization(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对彩色图像进行直方图均衡化 (HSV, YCrCb, RGB methods).

        自动计算并存储均衡化后的图像和它们的直方图。

        Returns:
            Tuple containing the equalized images: (hsv_equalized_rgb, ycrcb_equalized_rgb, rgb_equalized)
            The results are also stored in self.image_equalized and self.histogram_equalized.
        """

        if self.image is None or len(self.image.shape) != 3 or self.image.shape[2] != 3:
            raise ValueError("Input must be a 3-channel color image (RGB expected internally).")

        # Ensure image is uint8
        if self.image.dtype != np.uint8:
             # Try to scale common float representation
             if self.image.min() >= 0 and self.image.max() <= 1.0:
                 print("Warning: Converting float image to uint8 for color equalization.")
                 img_uint8 = (self.image * 255).astype(np.uint8)
             elif self.image.min() >=0 and self.image.max() <= 255:
                  print("Warning: Casting image to uint8 for color equalization.")
                  img_uint8 = self.image.astype(np.uint8)
             else:
                 raise TypeError(f"Color equalization requires uint8 input. Got {self.image.dtype} with range [{self.image.min()}, {self.image.max()}]")
        else:
            img_uint8 = self.image

        # Initialize storage for results
        # Use lists first, then convert to numpy array if needed, or just keep as list
        equalized_images_list = []
        equalized_histograms_list = []


        # --- HSV method ---
        print("Processing HSV equalization...")
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        # Equalize the V (Value/Brightness) channel
        v_channel = hsv[:, :, 2]
        if v_channel.size > 0: # Check if channel has data
            hsv[:, :, 2] = cv2.equalizeHist(v_channel)
        else:
            print("Warning: V channel is empty in HSV.")
        img_eq_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        equalized_images_list.append(img_eq_hsv)
        # Compute histogram for the resulting RGB image
        hist_hsv_eq = self.compute_histogram(img_eq_hsv, bins=self.bins, range_values=(0, self.bins))
        equalized_histograms_list.append(hist_hsv_eq)


        # --- YCrCb method ---
        print("Processing YCrCb equalization...")
        ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb)
        # Equalize the Y (Luma/Brightness) channel
        y_channel = ycrcb[:, :, 0]
        if y_channel.size > 0:
            ycrcb[:, :, 0] = cv2.equalizeHist(y_channel)
        else:
             print("Warning: Y channel is empty in YCrCb.")
        img_eq_ycrcb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        equalized_images_list.append(img_eq_ycrcb)
        # Compute histogram for the resulting RGB image
        hist_ycrcb_eq = self.compute_histogram(img_eq_ycrcb, bins=self.bins, range_values=(0, self.bins))
        equalized_histograms_list.append(hist_ycrcb_eq)

        # --- RGB method (Equalize each channel independently) ---
        print("Processing RGB equalization...")
        img_eq_rgb = np.zeros_like(img_uint8)
        for i in range(3):
            channel_data = img_uint8[:, :, i]
            if channel_data.size > 0:
                img_eq_rgb[:, :, i] = cv2.equalizeHist(channel_data)
            else:
                print(f"Warning: RGB Channel {i} is empty.")

        equalized_images_list.append(img_eq_rgb)
        # Compute histogram for the resulting RGB image
        hist_rgb_eq = self.compute_histogram(img_eq_rgb, bins=self.bins, range_values=(0, self.bins))
        equalized_histograms_list.append(hist_rgb_eq)

        # Store results in instance attributes
        # Use numpy arrays for easier indexing later if needed
        self.image_equalized = np.array(equalized_images_list)
        self.histogram_equalized = np.array(equalized_histograms_list)

        return tuple(equalized_images_list) # Return as a tuple for backward compatibility maybe? Or return the numpy array directly

# Example Usage (Add this outside the class if you want to run it)
if __name__ == '__main__':
    pipeline = HistogramEqualizationPipeline()

    # --- Test Grayscale ---
    try:
        # Create a dummy grayscale image file (replace with your actual image path)
        gray_test_path = 'test_gray.png'
        dummy_gray = np.random.randint(0, 256, (100, 150), dtype=np.uint8)
        # Add some contrast variation
        dummy_gray[20:60, 40:80] = np.clip(dummy_gray[20:60, 40:80] * 0.5, 0, 255).astype(np.uint8)
        dummy_gray[70:90, 100:140] = np.clip(dummy_gray[70:90, 100:140] * 1.5, 0, 255).astype(np.uint8)
        cv2.imwrite(gray_test_path, dummy_gray)

        print("\n--- Testing Grayscale Image ---")
        pipeline.load_image(gray_test_path, grayscale=True)
        # pipeline.load_image('path/to/your/grayscale_image.jpg', grayscale=True) # Use your image
        pipeline.display_image_with_histogram("Original Grayscale Image and Histogram")
        # pipeline.histogram_equalization(method='global')
        pipeline.histogram_equalization(method='local') # Test CLAHE
        pipeline.display_original_vs_equalized("Grayscale: Original vs Equalized (CLAHE)")
        pipeline.histogram_comparison() # Compare histograms

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred during grayscale testing: {e}")


    # --- Test Color ---
    try:
        # Create a dummy color image file (replace with your actual image path)
        color_test_path = 'test_color.png'
        dummy_color = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)
         # Add some color/contrast variation
        dummy_color[10:40, 20:50, 0] = np.clip(dummy_color[10:40, 20:50, 0] * 0.6, 0, 255).astype(np.uint8) # Less Red
        dummy_color[60:90, 80:120, 1] = np.clip(dummy_color[60:90, 80:120, 1] * 1.4, 0, 255).astype(np.uint8) # More Green
        cv2.imwrite(color_test_path, cv2.cvtColor(dummy_color, cv2.COLOR_RGB2BGR)) # Save as BGR

        print("\n--- Testing Color Image ---")
        pipeline.load_image(color_test_path, grayscale=False)
        # pipeline.load_image('path/to/your/color_image.jpg') # Use your image
        pipeline.display_image_with_histogram("Original Color Image and Histogram")
        pipeline.histogram_equalization(method='global') # Performs HSV, YCrCb, RGB equalization
        #pipeline.histogram_equalization(method='local') # Test color CLAHE (on V channel)
        pipeline.display_original_vs_equalized("Color: Original vs Equalized Methods")
        pipeline.histogram_comparison() # Compare histograms

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred during color testing: {e}")

    print("\nPipeline finished.")