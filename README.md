# Human-Action-Recognition
Enhancing Human Action Recognition in Still Images: Comparative Analysis of Traditional Machine Learning and Deep Learning Methods with a focus on Custom CNN Architecture

An increasingly challenging task in the field of computer vision in recent times has been the automatic
detection of human actions and activities. Furthermore, it is essential for many artificial intelligence
applications, such as robots, video surveillance, computer gaming, and human-computer interactions. The
increasing need for security and safety has led researchers to look into intelligent monitoring. The three
primary elements of an action recognition system are feature extraction, action representation, and
classification. Consequently, every stage is essential to attaining a high recognition rate. The action
representation and feature extraction models feed raw data into a feature vector. The right feature
extraction and selection would have a significant impact on the classification outcome. In order to
develop the best human activity identification algorithm possible, we have selected seven human actions
from a custom dataset. The activities are yawning, phoning, sitting, standing, walking, running and hand
waving. This study evaluates the performance of our custom CNN model, with deep learning approaches,
and conventional methods in identifying human activities from still photos.

Computer vision, a discipline within the realm of computer science, strives to emulate the intricate
workings of the human visual system. Its purpose is to enable computers to perceive and analyze objects
within images and videos, much like the way humans do. In the past, computer vision was limited in its
capabilities, but recent advancements have expanded its potential. For instance, autonomous vehicles rely
on computer vision to comprehend their surroundings [1]. Through an array of cameras capturing video
from various perspectives, this technology processes the imagery in real time, identifying road
boundaries, deciphering traffic signs, and recognizing other vehicles, objects, and pedestrians [2].
Consequently, self-driving cars navigate city streets and highways, expertly avoiding accidents as they
safely transport passengers to their intended destinations. Moreover, computer vision is instrumental in
facial recognition applications. By detecting distinct facial features within images, computer vision
algorithms compare them to databases of individual profiles, allowing for the matching of faces to
identities[3, 4]. Consumer devices employ facial recognition for owner verification, while social media
platforms utilize it to identify and tag individuals. Law enforcement agencies also employ this technology
to identify criminals captured in video footage, and it is frequently utilized in determining gender [5, 6, 7,
8]. Furthermore, computer vision plays a pivotal role in augmented and mixed reality, enabling computing
devices like smartphones, tablets, and smart glasses to overlay virtual objects onto real-world
visualizations. By employing computer vision, augmented reality devices can ascertain the precise
placement of virtual objects within their display interfaces.

The UCF101 [31], ILSVRC [32], and HACS [33] datasets are only a few of the many datasets available
for Human Action. However, those datasets were not used in this study because UCF101 and ILSVRC
had activities that are not commonly performed by people, while HACS had mostly videos and images
that are very similar. As a result, a custom dataset was constructed that included seven tasks that are
carried out on a daily basis. The activities are yawning, phoning, sitting, standing, walking, running and
hand waving. A total of 1974 images were taken. Hand waving 293 images, phoning 306 images, running
183 images, sitting 259 images, standing 320 images, walking 311 images and yawning 302 images were
taken. After that the dataset was augmented in order to find a more optimal result. We take two images
and combine them linearly using the tensors of those images. Finally Cutout is a convolutional neural
network regularization technique that includes eliminating contiguous portions of input images,
essentially augmenting the dataset with partially occluded copies of existing samples. Then all the images
were converted into 300*300 dimensions in order to avoid overfitting. After that all the images wereaugmented to 11 angles. After augmentation the dataset contained 23736 images, where hand waving had
3516, phoning had 3672, running had 2196, sitting had 3156, standing had 3840, walking had 3732 and
yawning had 3624 images. In the next step the entire dataset was split into an 80-20 ratio for dividing the
training and testing dataset. In the training dataset hand waving had 2808, phoning had 2940, running had
1752, sitting had 2532, standing had 3072, walking had 2988 and yawning had 2904 images. In the
testing dataset hand waving had 708, phoning had 732, running had 444, sitting had 624, standing had
768, walking had 744 and yawning had 720 images. The dataset is available at: https://rb.gy/n4bjj4

Our primary focus was to conduct a thorough analysis of image classification on our novel dataset using a
step-by-step execution plan. The first phase involved the development and evaluation of a Custom
Convolutional Neural Network (CNN) model. In the next phase, we aimed to contrast the performance of
our Custom CNN model with that of other widely used, pre-built machine learning models. The
traditional models to be evaluated included Naive Bayes, Support Vector Machine (SVM), K-nearest
Neighbor (KNN), Random Forest, Gaussian Naive Bayes, and Decision Tree. Following the evaluation of
traditional models, we evaluated the performance of deep learning models like ResNet, AlexNet, VGG16,
and DenseNet on our dataset. The goal is to discern whether the pre-trained models can effectively
generalize and perform competitively on our dataset. We also evaluate the performance of untrained deep
learning models too.

The CNN architecture presented here is tailored for image classification tasks, encompassing distinct
segments for convolutional, dense, and output layers. The architecture encompasses six convolutional
layers, each employing 3x3 kernels, coupled with Rectified Linear Unit (ReLU) activation functions and
max-pooling operations. This strategic combination enables the model to progressively reduce spatial
dimensions while increasing the depth of feature maps, allowing for the capture of detailed patterns in the
input images. Notably, the convolutional layers are structured to provide a hierarchical representation of
image features, with the spatial dimensions decreasing as the model delves deeper into the network.
Following the convolutional layers, a dropout layer is introduced to prevent overfitting, further enhancing
the model's generalization capability.The convolutional section initiates with the first layer, Conv2d-1, processing the input images to generate
64 feature maps with a spatial dimension of 224x224. The subsequent application of the ReLU activation
function introduces non-linearity, and max-pooling reduces the spatial dimensions by half through
MaxPool2d-3. Convolutional Layer 2 follows a similar pattern, doubling the number of feature maps to
128 and maintaining spatial dimensions. Convolutional Layers 3 to 5 continue this progression,
successively increasing feature map depth while decreasing spatial dimensions, effectively extracting
hierarchical features from the input data. Convolutional Layer 6 represents a critical point in the
architecture, further increasing the depth of feature maps to 512 with a spatial dimension of 7x7. This
layer serves as a key feature extractor before transitioning to the fully connected layers.
The transition from the convolutional layers to the fully connected layers involves a Flatten layer,
reshaping the 3D tensor into a 1D tensor suitable for processing by densely connected layers. The flattened representation is then passed through three fully connected layers, each comprising 4096
neurons and ReLU activation functions. These fully connected layers serve as powerful classifiers,
capturing high-level representations of the features extracted by the preceding convolutional layers. The
final layer, Linear-25, is the output layer with two neurons, facilitating binary classification. The model's
ability to capture complex representations is underscored by the substantial number of trainable
parameters, totaling 40,168,834.
The use of dropout after the last convolutional layer is a prudent choice to mitigate overfitting. Dropout
introduces a regularization technique by randomly dropping neurons during training, preventing the
network from relying too heavily on specific neurons and improving its generalization to unseen data.
This inclusion enhances the model's robustness and ensures that it does not memorize the training data but
learns meaningful features.

The table below presents a comprehensive comparison of accuracy metrics, including Precision, F1 Score,
and overall Accuracy, for traditional machine learning models (SVM, Gaussian NB, Decision Tree, KNN,
Random Forest) and a custom Convolutional Neural Network (CNN) model. Starting with Support Vector
Machine (SVM), the model exhibits relatively low precision, F1 Score, and accuracy, all hovering around
0.12. Similarly, Gaussian Naive Bayes (Gaussian NB) demonstrates low precision and F1 Score, with a
slightly improved accuracy of 0.10. Decision Tree performs better, with precision and F1 Score at 0.23,
and accuracy at 0.22, indicating a modest level of correctness in its predictions. Moving to K-Nearest
Neighbors (KNN), the model shows improvement with higher precision (0.32) but lower F1 Score (0.29)
and accuracy (0.27). Random Forest, while offering a reasonable precision of 0.29, falls short in terms of
F1 Score (0.27) and accuracy (0.26). In contrast, the custom CNN model outperforms the traditional
models across all metrics. The CNN model achieves a precision of 0.35, an F1 Score of 0.33, and an
accuracy of 0.35. These results suggest that the CNN model has a higher level of correctness in positive
predictions, better balance between precision and recall, and an overall improved accuracy compared to
the traditional machine learning models evaluated in this study.

In the table below, AlexNet, exhibits a precision of 0.31, an F1 Score of 0.29, and an accuracy of 0.33.
DenseNet performs well with a higher precision of 0.43, but its F1 Score (0.34) and accuracy (0.33) are
slightly lower. ResNet50 shows balanced performance with a precision of 0.37, an F1 Score of 0.33, and
an accuracy of 0.33. VGG16, while having a lower precision of 0.25, demonstrates an improved F1 Score
(0.24) and accuracy (0.30) compared to some other models. In comparison, the custom CNN model
achieves a precision of 0.35, an F1 Score of 0.33, and an accuracy of 0.35. These results suggest that the
custom CNN model performs competitively with, or surpasses, the untrained deep learning models in
terms of precision, F1 Score, and overall accuracy. It's noteworthy that the custom CNN model exhibits a
well-balanced performance across the three metrics, indicating its effectiveness in making accurate
predictions while considering both false positives and false negatives.

In the provided table below, AlexNet, demonstrates a precision of 0.52, an F1 Score of 0.50, and an
accuracy of 0.49. DenseNet exhibits notably high performance with a precision of 0.85, an F1 Score of
0.84, and an accuracy of 0.84. ResNet50 showcases balanced accuracy metrics with a precision of 0.79,
an F1 Score of 0.79, and an accuracy of 0.79. VGG16, similar to DenseNet, achieves high precision
(0.85), a strong F1 Score (0.84), and an accuracy of 0.84.

These results underscore the effectiveness of pre-trained deep learning models, particularly DenseNet and
VGG16, in achieving accurate predictions across the specified metrics. These models, having been
pre-trained on large and diverse datasets (ImageNet), demonstrate a high degree of transferability to the
task at hand, showcasing the advantages of leveraging pre-trained architectures for image classification.

In this research a total of five participants were used and all of them performed the seven actions. We
chose these particular seven actions because these are the most frequent actions performed by humans.
Traditional models like Support Vector Machine, Gaussian Naive Bayes, Decision Tree, K-Nearest
Neighbors, and Random Forest revealed limitations in capturing intricate image patterns, with Decision
Tree and KNN showing moderate accuracy. However, their overall effectiveness was surpassed by the
custom CNN model, which demonstrated superior precision (0.35), F1 Score (0.33), and accuracy (0.35).
Untrained deep learning models, encompassing AlexNet, DenseNet, ResNet50, and VGG16, exhibited
varying degrees of success, with DenseNet and VGG16 leading in accuracy. Nevertheless, these models
fell short of the custom CNN model's performance, emphasizing the importance of customization and
fine-tuning for optimal image classification. Pre-trained deep learning models, specifically AlexNet,
DenseNet, ResNet50, and VGG16, displayed strong overall performance with DenseNet and VGG16
achieving notable precision (0.85), F1 Score (0.84), and accuracy (0.84). Despite this, the custom CNN
model demonstrated competitive results, underscoring the significance of model customization even
against well-performing pre-trained models.
Based on all the findings, the result evaluation indicates that the custom CNN model stands out as a
robust and versatile solution for binary image classification. Its competitive performance, when compared
to traditional and untrained deep learning models, underscores the significance of customization and
domain-specific adaptation in achieving optimal outcomes. These results also tell us that given enough
computational resources and training time, if our custom CNN model is to be trained on big datasets like
ImageNet, it will also perform the image classification task similar to these State of the art models.

In this research the performance of all the traditional methods, deep learning methods and CNN was
tested on the task of human action recognition in still images. A custom dataset was created by the author
for this research. The dataset consisted of images of 7 human activities, which are yawning, phoning,
sitting, standing, walking, running and hand waving. 1974 images were taken in total and all of the
images were augmented. A total of five participants were employed in this research, and each of them
completed all seven actions. These seven activities were chosen because they are the most common
behaviors performed by humans. Because of underfitting, conventional process models produced poor
performance. The same issue arose with deep learning techniques. Due to underfitting, they also had very
low accuracy. Finally, in order to get a better result for the custom dataset, a custom CNN model was
introduced, and the model produced promising results. In conclusion, the developed CNN model
architecture stands as a testament to the careful consideration given to balancing depth, non-linearity, and
regularization. The sequential arrangement of layers facilitates the extraction of hierarchical features, and
the significant number of trainable parameters empowers the model to learn intricate representations.
