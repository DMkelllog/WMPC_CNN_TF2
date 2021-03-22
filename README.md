# Wafer map pattern classification using CNN

Wafer map defect pattern classification using CNN

## Methodology

### Convolutional Neural Network

![](https://github.com/DMkelllog/Wafer_map_pattern_classification_CNN/blob/main/CNN%20flow.PNG?raw=true)

* Input:    wafer map
* Output: predicted class
* Model:  CNN (based on VGG16)

## Data

* WM811K
  * dataset provided by MIR Lab (http://mirlab.org/dataset/public/)
  * .pkl file downloaded from Kaggle dataset (https://www.kaggle.com/qingyi/wm811k-wafer-map)
  * directory: /data/LSWMD.pkl

## Dependencies

* Python
* Pandas
* Tensorflow
* Scikit-learn
* Scikit-image

## References

* Nakazawa, T., & Kulkarni, D. V. (2018). Wafer map defect pattern classification and image retrieval using convolutional neural network. IEEE Transactions on Semiconductor Manufacturing, 31(2), 309-314.
* Shim, J., Kang, S., & Cho, S. (2020). Active learning of convolutional neural network for cost-effective wafer map pattern classification. IEEE Transactions on Semiconductor Manufacturing, 33(2), 258-266.
* Kang, S. (2020). Rotation-Invariant Wafer Map Pattern Classification With Convolutional Neural Networks. IEEE Access, 8, 170650-170658.

