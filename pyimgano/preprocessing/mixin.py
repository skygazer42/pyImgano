"""
Preprocessing mixin for easy integration with detection models.

This mixin allows anomaly detection models to easily incorporate
preprocessing capabilities.
"""

import logging
from typing import List, Optional, Union

import cv2
import numpy as np
from numpy.typing import NDArray

from .enhancer import ImageEnhancer, PreprocessingPipeline

logger = logging.getLogger(__name__)


class PreprocessingMixin:
    """
    Mixin class to add preprocessing capabilities to detection models.

    This mixin provides:
    - Easy access to image enhancement operations
    - Preprocessing pipeline management
    - Automatic image loading and preprocessing

    Examples
    --------
    >>> class MyDetector(PreprocessingMixin, BaseDetector):
    ...     def __init__(self, **kwargs):
    ...         super().__init__(**kwargs)
    ...         self.setup_preprocessing()
    ...
    ...     def fit(self, X, y=None):
    ...         # Preprocess images before fitting
    ...         X_processed = self.preprocess_images(X)
    ...         # ... continue with fitting
    ...
    >>> detector = MyDetector()
    >>> detector.add_preprocessing_step('gaussian_blur', ksize=(5, 5))
    >>> detector.add_preprocessing_step('edge_detection', method='canny')
    >>> detector.fit(train_images)
    """

    def setup_preprocessing(
        self,
        enable: bool = True,
        use_pipeline: bool = False,
    ) -> None:
        """
        Setup preprocessing components.

        Parameters
        ----------
        enable : bool, default=True
            Whether to enable preprocessing
        use_pipeline : bool, default=False
            Whether to use pipeline mode (multiple sequential operations)

        Examples
        --------
        >>> detector.setup_preprocessing(enable=True, use_pipeline=True)
        """
        self.preprocessing_enabled = enable
        self.use_preprocessing_pipeline = use_pipeline

        if enable:
            self.enhancer = ImageEnhancer()
            if use_pipeline:
                self.preprocessing_pipeline = PreprocessingPipeline()
                logger.info("Preprocessing enabled with pipeline mode")
            else:
                logger.info("Preprocessing enabled")
        else:
            logger.info("Preprocessing disabled")

    def add_preprocessing_step(
        self,
        operation: str,
        **kwargs
    ) -> None:
        """
        Add a preprocessing step to the pipeline.

        Parameters
        ----------
        operation : str
            Operation name (e.g., 'gaussian_blur', 'edge_detection')
        **kwargs : dict
            Operation parameters

        Examples
        --------
        >>> detector.add_preprocessing_step('gaussian_blur', ksize=(5, 5))
        >>> detector.add_preprocessing_step('edge_detection', method='canny')
        >>> detector.add_preprocessing_step('normalize', method='minmax')
        """
        if not self.preprocessing_enabled:
            raise RuntimeError("Preprocessing not enabled. Call setup_preprocessing() first.")

        if not self.use_preprocessing_pipeline:
            raise RuntimeError("Pipeline mode not enabled. Set use_pipeline=True.")

        self.preprocessing_pipeline.add_step(operation, **kwargs)
        logger.info("Added preprocessing step: %s", operation)

    def preprocess_image(
        self,
        image: Union[str, NDArray],
        operation: Optional[str] = None,
        **kwargs
    ) -> NDArray:
        """
        Preprocess a single image.

        Parameters
        ----------
        image : str or ndarray
            Image path or image array
        operation : str, optional
            Single operation to apply (if not using pipeline)
        **kwargs : dict
            Operation parameters

        Returns
        -------
        processed : ndarray
            Processed image

        Examples
        --------
        >>> # Single operation
        >>> img_blur = detector.preprocess_image('image.jpg', 'gaussian_blur', ksize=(5, 5))
        >>>
        >>> # Using pipeline
        >>> detector.add_preprocessing_step('gaussian_blur', ksize=(5, 5))
        >>> detector.add_preprocessing_step('edge_detection', method='canny')
        >>> img_processed = detector.preprocess_image('image.jpg')
        """
        if not self.preprocessing_enabled:
            # Just load and return
            if isinstance(image, str):
                return cv2.imread(image)
            return image

        # Load image if path
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Failed to load image: {image}")

        # Apply preprocessing
        if self.use_preprocessing_pipeline and hasattr(self, 'preprocessing_pipeline'):
            # Use pipeline
            if len(self.preprocessing_pipeline) == 0:
                logger.warning("Preprocessing pipeline is empty")
                return image
            return self.preprocessing_pipeline.transform(image)

        elif operation is not None:
            # Apply single operation
            if not hasattr(self.enhancer, operation):
                raise ValueError(f"Unknown operation: {operation}")
            func = getattr(self.enhancer, operation)
            return func(image, **kwargs)

        else:
            # No preprocessing
            return image

    def preprocess_images(
        self,
        images: List[Union[str, NDArray]],
        operation: Optional[str] = None,
        **kwargs
    ) -> List[NDArray]:
        """
        Preprocess multiple images.

        Parameters
        ----------
        images : list
            List of image paths or arrays
        operation : str, optional
            Single operation to apply
        **kwargs : dict
            Operation parameters

        Returns
        -------
        processed : list of ndarray
            Processed images

        Examples
        --------
        >>> images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
        >>> processed = detector.preprocess_images(images)
        """
        return [
            self.preprocess_image(img, operation, **kwargs)
            for img in images
        ]

    # Convenience methods for common operations

    def preprocess_with_edges(
        self,
        image: Union[str, NDArray],
        method: str = "canny",
        **kwargs
    ) -> NDArray:
        """
        Preprocess image with edge detection.

        Parameters
        ----------
        image : str or ndarray
            Input image
        method : str, default='canny'
            Edge detection method
        **kwargs : dict
            Method parameters

        Returns
        -------
        edges : ndarray
            Edge-detected image
        """
        if not self.preprocessing_enabled:
            raise RuntimeError("Preprocessing not enabled")

        if isinstance(image, str):
            image = cv2.imread(image)

        return self.enhancer.detect_edges(image, method, **kwargs)

    def preprocess_with_blur(
        self,
        image: Union[str, NDArray],
        filter_type: str = "gaussian",
        **kwargs
    ) -> NDArray:
        """
        Preprocess image with blur filter.

        Parameters
        ----------
        image : str or ndarray
            Input image
        filter_type : str, default='gaussian'
            Filter type ('gaussian', 'bilateral', 'median')
        **kwargs : dict
            Filter parameters

        Returns
        -------
        blurred : ndarray
            Blurred image
        """
        if not self.preprocessing_enabled:
            raise RuntimeError("Preprocessing not enabled")

        if isinstance(image, str):
            image = cv2.imread(image)

        if filter_type == "gaussian":
            return self.enhancer.gaussian_blur(image, **kwargs)
        elif filter_type == "bilateral":
            return self.enhancer.bilateral_filter(image, **kwargs)
        elif filter_type == "median":
            return self.enhancer.median_blur(image, **kwargs)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

    def preprocess_with_morphology(
        self,
        image: Union[str, NDArray],
        operation: str = "erosion",
        **kwargs
    ) -> NDArray:
        """
        Preprocess image with morphological operation.

        Parameters
        ----------
        image : str or ndarray
            Input image
        operation : str, default='erosion'
            Operation ('erosion', 'dilation', 'opening', 'closing')
        **kwargs : dict
            Operation parameters

        Returns
        -------
        result : ndarray
            Processed image
        """
        if not self.preprocessing_enabled:
            raise RuntimeError("Preprocessing not enabled")

        if isinstance(image, str):
            image = cv2.imread(image)

        operations_map = {
            'erosion': self.enhancer.erode,
            'dilation': self.enhancer.dilate,
            'opening': self.enhancer.opening,
            'closing': self.enhancer.closing,
            'gradient': self.enhancer.morphological_gradient,
        }

        if operation not in operations_map:
            raise ValueError(f"Unknown operation: {operation}")

        return operations_map[operation](image, **kwargs)

    def clear_preprocessing_pipeline(self) -> None:
        """Clear all preprocessing steps."""
        if not self.use_preprocessing_pipeline:
            raise RuntimeError("Pipeline mode not enabled")

        self.preprocessing_pipeline.clear()
        logger.info("Cleared preprocessing pipeline")

    def get_preprocessing_info(self) -> dict:
        """
        Get information about current preprocessing configuration.

        Returns
        -------
        info : dict
            Preprocessing configuration info
        """
        info = {
            'enabled': self.preprocessing_enabled,
            'pipeline_mode': self.use_preprocessing_pipeline,
        }

        if self.preprocessing_enabled and self.use_preprocessing_pipeline:
            info['pipeline_steps'] = len(self.preprocessing_pipeline)
            info['steps'] = [name for name, _, _ in self.preprocessing_pipeline.steps]

        return info


# Example usage in a detector
class ExampleDetectorWithPreprocessing(PreprocessingMixin):
    """
    Example detector showing how to use PreprocessingMixin.

    Examples
    --------
    >>> detector = ExampleDetectorWithPreprocessing()
    >>> detector.setup_preprocessing(enable=True, use_pipeline=True)
    >>> detector.add_preprocessing_step('gaussian_blur', ksize=(5, 5))
    >>> detector.add_preprocessing_step('normalize', method='minmax')
    >>> detector.fit(train_images)
    """

    def __init__(self):
        """Initialize detector."""
        # Initialize preprocessing
        self.setup_preprocessing(enable=True, use_pipeline=True)

    def fit(self, X, y=None):
        """
        Fit detector with preprocessing.

        Parameters
        ----------
        X : list
            Training images (paths or arrays)
        y : array-like, optional
            Labels (ignored for unsupervised)

        Returns
        -------
        self
        """
        # Preprocess all training images
        X_processed = self.preprocess_images(X)

        # Your fitting logic here
        logger.info("Fitting on %d preprocessed images", len(X_processed))

        return self

    def predict(self, X):
        """
        Predict with preprocessing.

        Parameters
        ----------
        X : list
            Test images (paths or arrays)

        Returns
        -------
        scores : ndarray
            Anomaly scores
        """
        # Preprocess all test images
        X_processed = self.preprocess_images(X)

        # Your prediction logic here
        scores = np.random.rand(len(X_processed))  # Placeholder

        return scores
