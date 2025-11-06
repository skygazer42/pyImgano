Detectors API Reference
=======================

This page provides detailed API documentation for all anomaly detection algorithms in PyImgAno.

Base Classes
------------

.. automodule:: pyimgano.detectors.base
   :members:
   :undoc-members:
   :show-inheritance:

Statistical Detectors
---------------------

IQR Detector
~~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.IQRDetector
   :members:
   :undoc-members:
   :show-inheritance:

MAD Detector
~~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.MADDetector
   :members:
   :undoc-members:
   :show-inheritance:

Z-Score Detector
~~~~~~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.ZScoreDetector
   :members:
   :undoc-members:
   :show-inheritance:

Histogram-based Detector
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.HistogramBasedDetector
   :members:
   :undoc-members:
   :show-inheritance:

Distance-based Detectors
------------------------

KNN Detector
~~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.KNNDetector
   :members:
   :undoc-members:
   :show-inheritance:

LOF Detector
~~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.LOFDetector
   :members:
   :undoc-members:
   :show-inheritance:

COF Detector
~~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.COFDetector
   :members:
   :undoc-members:
   :show-inheritance:

LOCI Detector
~~~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.LOCIDetector
   :members:
   :undoc-members:
   :show-inheritance:

Density-based Detectors
-----------------------

ECOD Detector
~~~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.ECODDetector
   :members:
   :undoc-members:
   :show-inheritance:

COPOD Detector
~~~~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.COPODDetector
   :members:
   :undoc-members:
   :show-inheritance:

Gaussian Mixture Model
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.GMMDetector
   :members:
   :undoc-members:
   :show-inheritance:

Kernel Density Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.KDEDetector
   :members:
   :undoc-members:
   :show-inheritance:

Isolation-based Detectors
-------------------------

Isolation Forest
~~~~~~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.IsolationForestDetector
   :members:
   :undoc-members:
   :show-inheritance:

Extended Isolation Forest
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.ExtendedIsolationForestDetector
   :members:
   :undoc-members:
   :show-inheritance:

Deep Learning Detectors
-----------------------

Autoencoder
~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.AutoencoderDetector
   :members:
   :undoc-members:
   :show-inheritance:

Variational Autoencoder (VAE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.VAEDetector
   :members:
   :undoc-members:
   :show-inheritance:

Deep SVDD
~~~~~~~~~

.. autoclass:: pyimgano.detectors.DeepSVDDDetector
   :members:
   :undoc-members:
   :show-inheritance:

DAGMM
~~~~~

.. autoclass:: pyimgano.detectors.DAGMMDetector
   :members:
   :undoc-members:
   :show-inheritance:

Memory-augmented Autoencoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.MemAEDetector
   :members:
   :undoc-members:
   :show-inheritance:

Reconstruction-based Detectors
------------------------------

PCA Detector
~~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.PCADetector
   :members:
   :undoc-members:
   :show-inheritance:

Kernel PCA Detector
~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.KernelPCADetector
   :members:
   :undoc-members:
   :show-inheritance:

Robust PCA Detector
~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.RobustPCADetector
   :members:
   :undoc-members:
   :show-inheritance:

One-Class Methods
-----------------

One-Class SVM
~~~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.OCSVMDetector
   :members:
   :undoc-members:
   :show-inheritance:

Ensemble Methods
----------------

Feature Bagging
~~~~~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.FeatureBaggingDetector
   :members:
   :undoc-members:
   :show-inheritance:

LSCP (Locally Selective Combination)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyimgano.detectors.LSCPDetector
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

.. automodule:: pyimgano.detectors.utils
   :members:
   :undoc-members:

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from pyimgano.detectors import IsolationForestDetector
   import numpy as np

   # Create detector
   detector = IsolationForestDetector(
       n_estimators=100,
       max_samples='auto',
       contamination=0.1
   )

   # Train
   X_train = np.random.randn(1000, 50)
   detector.fit(X_train)

   # Predict
   X_test = np.random.randn(100, 50)
   scores = detector.predict_proba(X_test)
   predictions = detector.predict(X_test)

Deep Learning Usage
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyimgano.detectors import AutoencoderDetector

   # Create detector
   detector = AutoencoderDetector(
       input_dim=784,  # e.g., 28x28 image flattened
       encoding_dim=32,
       hidden_dims=[256, 128],
       epochs=50,
       batch_size=32,
       learning_rate=0.001
   )

   # Train
   detector.fit(X_train)

   # Predict
   scores = detector.predict_proba(X_test)

Saving and Loading Models
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Save trained model
   detector.save_model('my_detector.pkl')

   # Load model
   from pyimgano.detectors import IsolationForestDetector
   loaded_detector = IsolationForestDetector.load_model('my_detector.pkl')

   # Use loaded model
   scores = loaded_detector.predict_proba(X_test)
