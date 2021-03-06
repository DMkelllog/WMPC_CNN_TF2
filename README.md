# Wafer map pattern classification using CNN

Wafer map defect pattern classification using CNN

## Methodology

### Convolutional Neural Network

![](https://github.com/DMkelllog/Wafer_map_pattern_classification_CNN/blob/main/CNN%20flow.PNG?raw=true)

* Input:    wafer map
  * resized to 64x64
* Output: predicted class
* Model:  CNN (based on VGG16)

## Data

* WM811K
  * 811457 wafer maps collected from 46393 lots in real-world fabrication

  * 172950 wafers were labeled by domain experts.

  * 9 defect classes (Center, Donut, Edge-ring, Edge-local, Local, Random, Near-full, Scratch, None)

    

  * provided by MIR Lab (http://mirlab.org/dataset/public/)

  * .pkl file downloaded from Kaggle dataset (https://www.kaggle.com/qingyi/wm811k-wafer-map)

  * directory: /data/LSWMD.pkl

## Dependencies

* Python
* Pandas
* Tensorflow
* Scikit-learn
* Scikit-image

## References

* WM-811K(LSWMD). National Taiwan University Department of Computer Science Multimedia Information Retrieval LAB http://mirlab.org/dataSet/public/
* Nakazawa, T., & Kulkarni, D. V. (2018). Wafer map defect pattern classification and image retrieval using convolutional neural network. IEEE Transactions on Semiconductor Manufacturing, 31(2), 309-314.
* Shim, J., Kang, S., & Cho, S. (2020). Active learning of convolutional neural network for cost-effective wafer map pattern classification. IEEE Transactions on Semiconductor Manufacturing, 33(2), 258-266.
* Kang, S. (2020). Rotation-Invariant Wafer Map Pattern Classification With Convolutional Neural Networks. IEEE Access, 8, 170650-170658.

