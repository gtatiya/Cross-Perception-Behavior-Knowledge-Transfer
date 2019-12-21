# Transferring Sensorimotor Cross-Perception and Cross-Behavior Knowledge for Object Categorization

**Abstract:**

> From an early stage in development, human infants learn to use exploratory behaviors to learn about the objects around them. Such exploration provides observations of how objects feel, sound, look, and move as a result of actions applied on them. Research in robotics has shown that robots too can use such behaviors (e.g., grasping, pushing, shaking) to infer object properties that cannot always be detected using visual input alone. Such learned representations are specific to each individual robot and cannot currently be transferred directly to another robot with different sensors and actions. Moreover, sensor failure can cause a robot to lose a specific sensory modality which may prevent it from using perceptual models that require it as input. To address these limitations, we propose a framework for knowledge transfer across behaviors and sensory modalities such that: 1) knowledge can be transferred from one or more robots to another, and, 2) knowledge can be transferred from one or more sensory modalities to another. We propose two different models for transfer based on variational auto-encoders and encoder-decoder networks. The intuition behind our approach is that if robots interact with a shared set of objects, the produced sensory data can be used to learn a mapping between multiple different feature spaces, each corresponding to a particular behavior coupled with a sensory modality. We evaluate our approach on a category recognition task using a dataset containing 9 robot behaviors, coupled with 4 sensory modalities, performed multiple times on a set of 100 objects. The results show that sensorimotor knowledge about objects can be transferred both across behaviors and across sensory modalities, such that a new robot (or the same robot, but with a different set of sensors) can bootstrap its category recognition models without having to exhaustively explore the full set of objects.

## Development Environment
For our research, we used Tufts High Performance Computing (HPC) that has NVIDIA Tesla P100 (16GB, 3584 CUDA Cores).
The neural networks were implemented in widely used deep learning framework `TensorFlow 1.12` with GPU support (CUDA 10.2).

## Dependencies

`Python 3.5.6` is used for development and following packages are required to run the code:<br><br>
`pip install tensorflow-gpu==1.12.0`<br>
`pip install sklearn==0.20.0`<br>
`pip install matplotlib==3.0.0`<br>
`pip install numpy==1.15.3`

## [Dataset](Datasets)

- [Visualization of each modalities](DatasetVisualization.ipynb)

