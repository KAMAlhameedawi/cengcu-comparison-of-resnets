# CIFAR-100 Image Classification using Deep Learning

Welcome to the repository for CIFAR-100 image classification using a variety of deep learning models. Our implementation includes a Linear Classifier, Convolutional Neural Net (CNNs) built from scratch, Residual Neural Nets (ResNet) also built from scratch, and Pretrained ResNet-18 and ResNet-50 models.

## Requirements and How to Use

To run the code, make sure you have the following prerequisites:

- Anaconda
- Pytorch

To execute the code directly from the GitHub repo:

1. Download the repository (ensure all requirements are met).
2. Run the code using Python 3.6 or on Google Colab GPU.

The dataset will be automatically downloaded from the torchvision.datasets.CIFAR100 library.

## Files

Here's a brief overview of the purpose of each Python file:

- **Linear_Classifier:** Implements an Artificial Neural Net with two hidden layers.
- **CNN:** Features a four-layered CNN built from scratch.
- **RESNET:** Includes a custom ResNet with two residual blocks.
- **PRETRAINED_RESNET:** Showcases pretrained ResNet-18 and ResNet-50 architectures fine-tuned on the CIFAR-100 dataset.

## Key Learnings

This project provided valuable insights, including:

- Building an ANN and a Convolutional NN from scratch.
- Crafting custom ResNet architectures from scratch.
- Leveraging pretraining and transfer learning techniques.
- Identifying optimal learning rates.
- Grasping learning schedulers such as OneCycleLR and CosineAnnealingLR and their applications.
- Using different optimizers such as SGD and ADAM.
- Applying various Data Augmentation techniques like random crop, random split, and random rotate.
- Implementing regularization methods such as L1 and L2 regularization, dropout, and early stopping.

##  Image Classification with ResNet and Transfer Learning ResNet-50

### Objective

The main goal of this homework is to gain hands-on experience with image classification using deep neural networks, specifically ResNet, through the application of transfer learning. Students will fine-tune pre-trained ResNet models on the CIFAR-100 dataset and document their results.

### Instructions

#### Dataset Selection

For this assignment, we have chosen the CIFAR-100 dataset. It comprises 100 classes, each containing 600 images, and the size of each image is 32x32 pixels.

#### Understand ResNet

ResNet, or Residual Networks, is a deep learning architecture that introduces the concept of residual learning. It addresses the challenges of training very deep neural networks by incorporating skip connections and residual blocks. ResNet facilitates the training of deeper networks, mitigating the vanishing gradient problem.

#### Pre-trained ResNet Models

We opted for ResNet-50 and found a pre-trained model using PyTorch. The pre-trained model is loaded, and its architecture is explored.

#### Fine-Tuning

The fine-tuning process involves adjusting the final layers of the pre-trained ResNet-50 model to match the number of classes in the CIFAR-100 dataset. The model is then trained on the training set and validated on the validation set.

#### Experiments

Various experiments are conducted with different hyperparameters, including learning rates, batch sizes, and training epochs. The results, including accuracy and loss metrics, are reported for each experiment.

#### Performance Evaluation

The performance of the fine-tuned ResNet-50 model is evaluated on the test set of the CIFAR-100 dataset. A discussion is provided regarding the achieved accuracy and observations regarding the model's performance.

## Result 

Downloading https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz to ./data/cifar-100-python.tar.gz

custom_ResNet(
  (conv_layer_1): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (conv_layer_2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (res_layer1): Sequential(
    (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
  )
  (conv_layer_3): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv_layer_4): Sequential(
    (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (res_layer2): Sequential(
    (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
  )
  (classifier): Sequential(
    (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (1): Flatten(start_dim=1, end_dim=-1)
    (2): Linear(in_features=2048, out_features=100, bias=True)
  )
)


 ## Training 

 Adjusting learning rate of group 0 to 1.0000e-01.
Adjusting learning rate of group 0 to 9.0451e-02.
Number of epochs: 0 | Validation loss : 3.585869073867798  | Training loss : 3.6891467571258545  |   Training accuracy: 0.1304749995470047 validation accuracy : 0.16620223224163055
Adjusting learning rate of group 0 to 6.5451e-02.
Number of epochs: 1 | Validation loss : 3.1567494869232178  | Training loss : 3.1001322269439697  |   Training accuracy: 0.23407499492168427 validation accuracy : 0.2447253167629242
Adjusting learning rate of group 0 to 3.4549e-02.
Number of epochs: 2 | Validation loss : 2.691833972930908  | Training loss : 2.5806643962860107  |   Training accuracy: 0.33922499418258667 validation accuracy : 0.32016322016716003
Adjusting learning rate of group 0 to 9.5492e-03.
Number of epochs: 3 | Validation loss : 2.382085084915161  | Training loss : 2.0880346298217773  |   Training accuracy: 0.447299987077713 validation accuracy : 0.3905254900455475
Adjusting learning rate of group 0 to 0.0000e+00.
Number of epochs: 4 | Validation loss : 2.141857385635376  | Training loss : 1.671291708946228  |   Training accuracy: 0.5479750037193298 validation accuracy : 0.44735270738601685
Adjusting learning rate of group 0 to 9.5492e-03.
Number of epochs: 5 | Validation loss : 2.0969603061676025  | Training loss : 1.5359559059143066  |   Training accuracy: 0.5803999900817871 validation accuracy : 0.45680731534957886
Adjusting learning rate of group 0 to 3.4549e-02.
Number of epochs: 6 | Validation loss : 2.1255850791931152  | Training loss : 1.5089224576950073  |   Training accuracy: 0.5903499722480774 validation accuracy : 0.4553144872188568
Adjusting learning rate of group 0 to 6.5451e-02.
Number of epochs: 7 | Validation loss : 2.31904935836792  | Training loss : 1.701501488685608  |   Training accuracy: 0.5337499976158142 validation accuracy : 0.42028263211250305
Adjusting learning rate of group 0 to 9.0451e-02.
Number of epochs: 8 | Validation loss : 2.4068708419799805  | Training loss : 1.7407087087631226  |   Training accuracy: 0.521049976348877 validation accuracy : 0.40266719460487366
Adjusting learning rate of group 0 to 1.0000e-01.
Number of epochs: 9 | Validation loss : 2.4864754676818848  | Training loss : 1.535573959350586  |   Training accuracy: 0.572700023651123 validation accuracy : 0.41301751136779785
Adjusting learning rate of group 0 to 1.0000e-02.
Adjusting learning rate of group 0 to 9.0451e-03.
Number of epochs: 0 | Validation loss : 1.8544304370880127  | Training loss : 0.7845197916030884  |   Training accuracy: 0.7820000052452087 validation accuracy : 0.540207028388977
Adjusting learning rate of group 0 to 6.5451e-03.
Number of epochs: 1 | Validation loss : 1.8322879076004028  | Training loss : 0.5390940308570862  |   Training accuracy: 0.8635500073432922 validation accuracy : 0.5450835824012756
Adjusting learning rate of group 0 to 3.4549e-03.
Number of epochs: 2 | Validation loss : 1.8284873962402344  | Training loss : 0.402696818113327  |   Training accuracy: 0.9141499996185303 validation accuracy : 0.5476711988449097
Adjusting learning rate of group 0 to 9.5492e-04.
Number of epochs: 3 | Validation loss : 1.8227754831314087  | Training loss : 0.32358965277671814  |   Training accuracy: 0.9425749778747559 validation accuracy : 0.5524482727050781
Adjusting learning rate of group 0 to 0.0000e+00.
Number of epochs: 4 | Validation loss : 1.820753574371338  | Training loss : 0.2861107289791107  |   Training accuracy: 0.9542499780654907 validation accuracy : 0.5549362897872925
Adjusting learning rate of group 0 to 9.5492e-04.
Number of epochs: 5 | Validation loss : 1.8209660053253174  | Training loss : 0.2740066945552826  |   Training accuracy: 0.9574499726295471 validation accuracy : 0.5550358295440674
Adjusting learning rate of group 0 to 3.4549e-03.
Number of epochs: 6 | Validation loss : 1.821519374847412  | Training loss : 0.2749149203300476  |   Training accuracy: 0.9583749771118164 validation accuracy : 0.5560310482978821
Adjusting learning rate of group 0 to 6.5451e-03.
Number of epochs: 7 | Validation loss : 1.8279714584350586  | Training loss : 0.26912909746170044  |   Training accuracy: 0.9604499936103821 validation accuracy : 0.5555334687232971
Adjusting learning rate of group 0 to 9.0451e-03.
Number of epochs: 8 | Validation loss : 1.8410375118255615  | Training loss : 0.24677719175815582  |   Training accuracy: 0.9664499759674072 validation accuracy : 0.5507563948631287
Adjusting learning rate of group 0 to 1.0000e-02.
Number of epochs: 9 | Validation loss : 1.8584562540054321  | Training loss : 0.20671406388282776  |   Training accuracy: 0.9772999882698059 validation accuracy : 0.5531449317932129
Adjusting learning rate of group 0 to 1.0000e-03.
Adjusting learning rate of group 0 to 9.0451e-04.
Number of epochs: 0 | Validation loss : 1.8511962890625  | Training loss : 0.1573496311903  |   Training accuracy: 0.9881500005722046 validation accuracy : 0.5588176846504211
Adjusting learning rate of group 0 to 6.5451e-04.
Number of epochs: 1 | Validation loss : 1.853040337562561  | Training loss : 0.14785555005073547  |   Training accuracy: 0.9905250072479248 validation accuracy : 0.5599124431610107
Adjusting learning rate of group 0 to 3.4549e-04.
Number of epochs: 2 | Validation loss : 1.854537844657898  | Training loss : 0.14269083738327026  |   Training accuracy: 0.991474986076355 validation accuracy : 0.559414803981781
Adjusting learning rate of group 0 to 9.5492e-05.
Number of epochs: 3 | Validation loss : 1.8553465604782104  | Training loss : 0.139185830950737  |   Training accuracy: 0.9922500252723694 validation accuracy : 0.5590167045593262
Adjusting learning rate of group 0 to 0.0000e+00.
Number of epochs: 4 | Validation loss : 1.8555516004562378  | Training loss : 0.13716286420822144  |   Training accuracy: 0.9928249716758728 validation accuracy : 0.5593152642250061
Adjusting learning rate of group 0 to 9.5492e-05.
Number of epochs: 5 | Validation loss : 1.8555916547775269  | Training loss : 0.13651098310947418  |   Training accuracy: 0.9928249716758728 validation accuracy : 0.559215784072876
Adjusting learning rate of group 0 to 3.4549e-04.
Number of epochs: 6 | Validation loss : 1.8557249307632446  | Training loss : 0.13666637241840363  |   Training accuracy: 0.9927999973297119 validation accuracy : 0.5590167045593262
Adjusting learning rate of group 0 to 6.5451e-04.
Number of epochs: 7 | Validation loss : 1.856405258178711  | Training loss : 0.13654521107673645  |   Training accuracy: 0.9927250146865845 validation accuracy : 0.5591162443161011
Adjusting learning rate of group 0 to 9.0451e-04.
Number of epochs: 8 | Validation loss : 1.8574522733688354  | Training loss : 0.13536161184310913  |   Training accuracy: 0.992775022983551 validation accuracy : 0.5591162443161011
Adjusting learning rate of group 0 to 1.0000e-03.
Number of epochs: 9 | Validation loss : 1.8589649200439453  | Training loss : 0.13293762505054474  |   Training accuracy: 0.9932000041007996 validation accuracy : 0.559215784072876
Adjusting learning rate of group 0 to 1.0000e-04.
Adjusting learning rate of group 0 to 9.0451e-05.
Number of epochs: 0 | Validation loss : 1.8593796491622925  | Training loss : 0.12890248000621796  |   Training accuracy: 0.9939749836921692 validation accuracy : 0.5587181448936462
Adjusting learning rate of group 0 to 6.5451e-05.
Number of epochs: 1 | Validation loss : 1.8596181869506836  | Training loss : 0.12832148373126984  |   Training accuracy: 0.9941250085830688 validation accuracy : 0.5587181448936462
Adjusting learning rate of group 0 to 3.4549e-05.
Number of epochs: 2 | Validation loss : 1.8597968816757202  | Training loss : 0.1278499811887741  |   Training accuracy: 0.9941999912261963 validation accuracy : 0.5586186051368713
Adjusting learning rate of group 0 to 9.5492e-06.
Number of epochs: 3 | Validation loss : 1.8598936796188354  | Training loss : 0.12750649452209473  |   Training accuracy: 0.994225025177002 validation accuracy : 0.5588176846504211
Adjusting learning rate of group 0 to 0.0000e+00.
Number of epochs: 4 | Validation loss : 1.8599272966384888  | Training loss : 0.12731227278709412  |   Training accuracy: 0.9942499995231628 validation accuracy : 0.5587181448936462
Adjusting learning rate of group 0 to 9.5492e-06.
Number of epochs: 5 | Validation loss : 1.8599320650100708  | Training loss : 0.1272539347410202  |   Training accuracy: 0.9943000078201294 validation accuracy : 0.5587181448936462
Adjusting learning rate of group 0 to 3.4549e-05.
Number of epochs: 6 | Validation loss : 1.8599474430084229  | Training loss : 0.12727093696594238  |   Training accuracy: 0.9942749738693237 validation accuracy : 0.5587181448936462
Adjusting learning rate of group 0 to 6.5451e-05.
Number of epochs: 7 | Validation loss : 1.8600174188613892  | Training loss : 0.12727367877960205  |   Training accuracy: 0.9942499995231628 validation accuracy : 0.5588176846504211
Adjusting learning rate of group 0 to 9.0451e-05.
Number of epochs: 8 | Validation loss : 1.8601715564727783  | Training loss : 0.12718285620212555  |   Training accuracy: 0.9943000078201294 validation accuracy : 0.5588176846504211
Adjusting learning rate of group 0 to 1.0000e-04.
Number of epochs: 9 | Validation loss : 1.860390305519104  | Training loss : 0.12696261703968048  |   Training accuracy: 0.9943249821662903 validation accuracy : 0.5587181448936462
![image](https://github.com/KAMAlhameedawi/cengcu-comparison-of-resnets/assets/149914341/9c1a6440-b932-4ed4-a347-f4743ff2fbcc)
##### Graph for Training Loss/epoch for Different Learning Rates

![image](https://github.com/KAMAlhameedawi/cengcu-comparison-of-resnets/assets/149914341/1bc198da-dc86-46c2-b8eb-4dafed5f08f6)
##### Validation and Training Accuracy per epoch
![image](https://github.com/KAMAlhameedawi/cengcu-comparison-of-resnets/assets/149914341/1e5d8536-d202-479a-813c-715efb6e5362)

##### Validation and Training Accuracy per epoch

![image](https://github.com/KAMAlhameedawi/cengcu-comparison-of-resnets/assets/149914341/bcc07d16-3c0b-45df-86b1-6aa114cacbc4)

#### Testing Accuracy (without Data augmentation and Droput)

custom_ResNet(
  (conv_layer_1): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (conv_layer_2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (dropout_layer): Dropout(p=0.25, inplace=False)
  (res_layer1): Sequential(
    (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
  )
  (conv_layer_3): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv_layer_4): Sequential(
    (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (res_layer2): Sequential(
    (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
  )
  (classifier): Sequential(
    (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (1): Flatten(start_dim=1, end_dim=-1)
    (2): Linear(in_features=2048, out_features=100, bias=True)
  )
)


Files already downloaded and verified
Files already downloaded and verified
40000
10000
Number of epochs: 0 | Validation loss : 4.268755912780762  | Training loss : 5.99457311630249  |   Training accuracy: 0.02734999917447567 validation accuracy : 0.04562699794769287
Number of epochs: 1 | Validation loss : 3.8958728313446045  | Training loss : 4.0522637367248535  |   Training accuracy: 0.08192499727010727 validation accuracy : 0.10043929517269135
Number of epochs: 2 | Validation loss : 3.6166913509368896  | Training loss : 3.7246155738830566  |   Training accuracy: 0.13552500307559967 validation accuracy : 0.14586661756038666
Number of epochs: 3 | Validation loss : 3.3115131855010986  | Training loss : 3.447417974472046  |   Training accuracy: 0.1815750002861023 validation accuracy : 0.19588658213615417
Number of epochs: 4 | Validation loss : 3.090261936187744  | Training loss : 3.196941614151001  |   Training accuracy: 0.22577500343322754 validation accuracy : 0.24101437628269196
Number of epochs: 5 | Validation loss : 2.785106897354126  | Training loss : 2.9471001625061035  |   Training accuracy: 0.27094998955726624 validation accuracy : 0.3020167648792267
Number of epochs: 6 | Validation loss : 2.6600756645202637  | Training loss : 2.723175048828125  |   Training accuracy: 0.31027498841285706 validation accuracy : 0.32288339734077454
Number of epochs: 7 | Validation loss : 2.4543557167053223  | Training loss : 2.5428543090820312  |   Training accuracy: 0.3493500053882599 validation accuracy : 0.35652956366539
Number of epochs: 8 | Validation loss : 2.383197069168091  | Training loss : 2.4028894901275635  |   Training accuracy: 0.376675009727478 validation accuracy : 0.3836860954761505
Number of epochs: 9 | Validation loss : 2.271010398864746  | Training loss : 2.2997093200683594  |   Training accuracy: 0.40095001459121704 validation accuracy : 0.4030551016330719
Number of epochs: 10 | Validation loss : 2.220926523208618  | Training loss : 2.182345151901245  |   Training accuracy: 0.4246250092983246 validation accuracy : 0.41573482751846313
Number of epochs: 11 | Validation loss : 2.149785280227661  | Training loss : 2.0915496349334717  |   Training accuracy: 0.44327500462532043 validation accuracy : 0.4344049394130707
Number of epochs: 12 | Validation loss : 2.141810655593872  | Training loss : 1.999045491218567  |   Training accuracy: 0.4670250117778778 validation accuracy : 0.4374001622200012
Number of epochs: 13 | Validation loss : 2.0197789669036865  | Training loss : 1.920047402381897  |   Training accuracy: 0.48477500677108765 validation accuracy : 0.4618610143661499
Number of epochs: 14 | Validation loss : 1.9755831956863403  | Training loss : 1.8561536073684692  |   Training accuracy: 0.4968999922275543 validation accuracy : 0.4747404158115387
Number of epochs: 15 | Validation loss : 1.9644920825958252  | Training loss : 1.7864583730697632  |   Training accuracy: 0.5134750008583069 validation accuracy : 0.48003193736076355
Number of epochs: 16 | Validation loss : 1.923583745956421  | Training loss : 1.7223565578460693  |   Training accuracy: 0.5258499979972839 validation accuracy : 0.48043131828308105
Number of epochs: 17 | Validation loss : 1.9482614994049072  | Training loss : 1.6710842847824097  |   Training accuracy: 0.5391749739646912 validation accuracy : 0.47863417863845825
Number of epochs: 18 | Validation loss : 1.8164079189300537  | Training loss : 1.6302298307418823  |   Training accuracy: 0.5491999983787537 validation accuracy : 0.507388174533844
Number of epochs: 19 | Validation loss : 1.8506920337677002  | Training loss : 1.575788140296936  |   Training accuracy: 0.5640749931335449 validation accuracy : 0.5032947063446045
Number of epochs: 20 | Validation loss : 1.840518593788147  | Training loss : 1.525460124015808  |   Training accuracy: 0.5750749707221985 validation accuracy : 0.5079872012138367
Number of epochs: 21 | Validation loss : 1.7943613529205322  | Training loss : 1.4644570350646973  |   Training accuracy: 0.5874000191688538 validation accuracy : 0.5230631232261658
Number of epochs: 22 | Validation loss : 1.7442023754119873  | Training loss : 1.4298579692840576  |   Training accuracy: 0.5980250239372253 validation accuracy : 0.5259584784507751
Number of epochs: 23 | Validation loss : 1.7440208196640015  | Training loss : 1.396497368812561  |   Training accuracy: 0.604200005531311 validation accuracy : 0.5305511355400085
Number of epochs: 24 | Validation loss : 1.7498215436935425  | Training loss : 1.3445626497268677  |   Training accuracy: 0.6202250123023987 validation accuracy : 0.5320487022399902
Number of epochs: 25 | Validation loss : 1.7117502689361572  | Training loss : 1.308963418006897  |   Training accuracy: 0.6294749975204468 validation accuracy : 0.542931318283081
Number of epochs: 26 | Validation loss : 1.7112985849380493  | Training loss : 1.2714604139328003  |   Training accuracy: 0.6378250122070312 validation accuracy : 0.5389376878738403
Number of epochs: 27 | Validation loss : 1.7359919548034668  | Training loss : 1.233381748199463  |   Training accuracy: 0.644225001335144 validation accuracy : 0.5455271601676941
Number of epochs: 28 | Validation loss : 1.6821820735931396  | Training loss : 1.2017607688903809  |   Training accuracy: 0.6531999707221985 validation accuracy : 0.5540135502815247
Number of epochs: 29 | Validation loss : 1.6805236339569092  | Training loss : 1.152880072593689  |   Training accuracy: 0.6653000116348267 validation accuracy : 0.5578075051307678

##### Training and Validation Accuracy

![image](https://github.com/KAMAlhameedawi/cengcu-comparison-of-resnets/assets/149914341/af846243-0078-4f1e-8165-2f87473af3e4)

#### Testing Accuracy

Final Test Accuracy for ResNet :  0.5512180328369141

# ADVANCED (20%)
#### Testing with Resnet 18

ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=100, bias=True)
)

Number of epochs: 0 | Validation loss : 2.4699254035949707  | Training loss : 3.197343587875366  |   Training accuracy: 0.2342749983072281 validation accuracy : 0.3608226776123047
Number of epochs: 1 | Validation loss : 2.2589735984802246  | Training loss : 2.4496846199035645  |   Training accuracy: 0.36844998598098755 validation accuracy : 0.4033546447753906
Number of epochs: 2 | Validation loss : 2.084754705429077  | Training loss : 2.183049440383911  |   Training accuracy: 0.42272499203681946 validation accuracy : 0.44369009137153625
Number of epochs: 3 | Validation loss : 2.042905569076538  | Training loss : 1.9954320192337036  |   Training accuracy: 0.4666000008583069 validation accuracy : 0.45656949281692505
Number of epochs: 4 | Validation loss : 1.9902265071868896  | Training loss : 1.8556722402572632  |   Training accuracy: 0.4988749921321869 validation accuracy : 0.4715455174446106
Number of epochs: 5 | Validation loss : 1.982299566268921  | Training loss : 1.742040991783142  |   Training accuracy: 0.5217000246047974 validation accuracy : 0.48142972588539124
Number of epochs: 6 | Validation loss : 1.9519826173782349  | Training loss : 1.6462445259094238  |   Training accuracy: 0.5468000173568726 validation accuracy : 0.4908146858215332
Number of epochs: 7 | Validation loss : 1.962587594985962  | Training loss : 1.5528128147125244  |   Training accuracy: 0.5687500238418579 validation accuracy : 0.4912140667438507
Number of epochs: 8 | Validation loss : 1.9520680904388428  | Training loss : 1.4738515615463257  |   Training accuracy: 0.5888500213623047 validation accuracy : 0.49650558829307556
Number of epochs: 9 | Validation loss : 1.939808964729309  | Training loss : 1.4088771343231201  |   Training accuracy: 0.6018249988555908 validation accuracy : 0.5006988644599915
Number of epochs: 0 | Validation loss : 1.874517560005188  | Training loss : 1.1982789039611816  |   Training accuracy: 0.6657249927520752 validation accuracy : 0.5145766735076904
Number of epochs: 1 | Validation loss : 1.8307397365570068  | Training loss : 1.1270666122436523  |   Training accuracy: 0.6856499910354614 validation accuracy : 0.522164523601532
Number of epochs: 2 | Validation loss : 1.823468804359436  | Training loss : 1.0957401990890503  |   Training accuracy: 0.6928499937057495 validation accuracy : 0.5288538336753845
Number of epochs: 3 | Validation loss : 1.8255858421325684  | Training loss : 1.057088017463684  |   Training accuracy: 0.7027000188827515 validation accuracy : 0.5250598788261414
Number of epochs: 4 | Validation loss : 1.8260201215744019  | Training loss : 1.0370711088180542  |   Training accuracy: 0.7075499892234802 validation accuracy : 0.5248602032661438
Number of epochs: 5 | Validation loss : 1.8159772157669067  | Training loss : 1.0271960496902466  |   Training accuracy: 0.7122499942779541 validation accuracy : 0.5327476263046265
Number of epochs: 6 | Validation loss : 1.8127281665802002  | Training loss : 1.003485083580017  |   Training accuracy: 0.7205749750137329 validation accuracy : 0.5310503244400024
Number of epochs: 7 | Validation loss : 1.8152077198028564  | Training loss : 0.9896181225776672  |   Training accuracy: 0.7246000170707703 validation accuracy : 0.5269568562507629
Number of epochs: 8 | Validation loss : 1.810781478881836  | Training loss : 0.9739364981651306  |   Training accuracy: 0.7279000282287598 validation accuracy : 0.5324480533599854
Number of epochs: 9 | Validation loss : 1.8422468900680542  | Training loss : 0.9583733081817627  |   Training accuracy: 0.7297000288963318 validation accuracy : 0.5289536714553833
Number of epochs: 0 | Validation loss : 1.8292641639709473  | Training loss : 0.9314661622047424  |   Training accuracy: 0.7404999732971191 validation accuracy : 0.5240615010261536
Number of epochs: 1 | Validation loss : 1.8204362392425537  | Training loss : 0.9336920380592346  |   Training accuracy: 0.7412499785423279 validation accuracy : 0.532947301864624
Number of epochs: 2 | Validation loss : 1.813590407371521  | Training loss : 0.9262114763259888  |   Training accuracy: 0.7389749884605408 validation accuracy : 0.53125
Number of epochs: 3 | Validation loss : 1.823628306388855  | Training loss : 0.9227796792984009  |   Training accuracy: 0.7433000206947327 validation accuracy : 0.5308506488800049
Number of epochs: 4 | Validation loss : 1.8088204860687256  | Training loss : 0.9184167981147766  |   Training accuracy: 0.7445250153541565 validation accuracy : 0.5330471396446228
Number of epochs: 5 | Validation loss : 1.8062653541564941  | Training loss : 0.9198330044746399  |   Training accuracy: 0.7439249753952026 validation accuracy : 0.5304512977600098
Number of epochs: 6 | Validation loss : 1.8192200660705566  | Training loss : 0.9187827706336975  |   Training accuracy: 0.7457000017166138 validation accuracy : 0.5306509733200073
Number of epochs: 7 | Validation loss : 1.8182536363601685  | Training loss : 0.9131468534469604  |   Training accuracy: 0.7449749708175659 validation accuracy : 0.5281549692153931
Number of epochs: 8 | Validation loss : 1.8314536809921265  | Training loss : 0.9174026250839233  |   Training accuracy: 0.7430499792098999 validation accuracy : 0.5251597166061401
Number of epochs: 9 | Validation loss : 1.8124843835830688  | Training loss : 0.913352906703949  |   Training accuracy: 0.7451500296592712 validation accuracy : 0.5327476263046265
CPU times: user 36min 6s, sys: 6.59 s, total: 36min 13s
Wall time: 36min 41s

#### Visualization

### Validation loss

![image](https://github.com/KAMAlhameedawi/cengcu-comparison-of-resnets/assets/149914341/3b87046c-7e78-4104-9122-bf72d9968576)

### Training loss

![image](https://github.com/KAMAlhameedawi/cengcu-comparison-of-resnets/assets/149914341/2dfa3259-b759-4f2d-8e23-c567353e9864)


#### Training and Validation Accuracy

![image](https://github.com/KAMAlhameedawi/cengcu-comparison-of-resnets/assets/149914341/137d57f8-d591-4703-9c09-dbce9154f56b)


#### Testing Accuracy

Number of epochs: 0 | Validation loss : 2.0258710384368896  | Training loss : 1.1428731679916382  |   Training accuracy: 0.6736000180244446 validation accuracy : 0.5015974640846252
Number of epochs: 1 | Validation loss : 2.0347113609313965  | Training loss : 1.087228775024414  |   Training accuracy: 0.6884250044822693 validation accuracy : 0.5
Number of epochs: 2 | Validation loss : 2.0311617851257324  | Training loss : 1.0542583465576172  |   Training accuracy: 0.6973749995231628 validation accuracy : 0.5024960041046143
Number of epochs: 3 | Validation loss : 2.0545201301574707  | Training loss : 1.0059000253677368  |   Training accuracy: 0.7067000269889832 validation accuracy : 0.5019968152046204
Number of epochs: 4 | Validation loss : 2.0406408309936523  | Training loss : 0.9538673758506775  |   Training accuracy: 0.7246249914169312 validation accuracy : 0.5101836919784546
Number of epochs: 5 | Validation loss : 2.1461501121520996  | Training loss : 0.9071981906890869  |   Training accuracy: 0.7354249954223633 validation accuracy : 0.49970048666000366
Number of epochs: 6 | Validation loss : 2.148185968399048  | Training loss : 0.8674132823944092  |   Training accuracy: 0.7475500106811523 validation accuracy : 0.49890175461769104
Number of epochs: 7 | Validation loss : 2.1235783100128174  | Training loss : 0.8335657715797424  |   Training accuracy: 0.7560999989509583 validation accuracy : 0.5092851519584656
Number of epochs: 8 | Validation loss : 2.1646902561187744  | Training loss : 0.7881748080253601  |   Training accuracy: 0.7685750126838684 validation accuracy : 0.5024960041046143
Number of epochs: 9 | Validation loss : 2.2075295448303223  | Training loss : 0.7536813020706177  |   Training accuracy: 0.7767249941825867 validation accuracy : 0.506489634513855
0.5081868767738342
CPU times: user 12min 21s, sys: 2.35 s, total: 12min 24s
Wall time: 12min 26s
### Test Accuracy for ResNet 18

Test Accuracy is :  0.5038937926292419

# ADVANCED (20%)
## ResNet 50

ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (layer2): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (3): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (layer3): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (3): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (4): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (5): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (layer4): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=2048, out_features=100, bias=True)
)

### Training

Number of epochs: 0 | Validation loss : 2.354605197906494  | Training loss : 3.080040216445923  |   Training accuracy: 0.2551000118255615 validation accuracy : 0.3850838541984558
Number of epochs: 1 | Validation loss : 2.06229567527771  | Training loss : 2.237826108932495  |   Training accuracy: 0.4109250009059906 validation accuracy : 0.4536741077899933
Number of epochs: 2 | Validation loss : 1.9198222160339355  | Training loss : 1.9390640258789062  |   Training accuracy: 0.4768249988555908 validation accuracy : 0.4862220585346222
Number of epochs: 3 | Validation loss : 1.8678064346313477  | Training loss : 1.7458786964416504  |   Training accuracy: 0.5234249830245972 validation accuracy : 0.5043929815292358
Number of epochs: 4 | Validation loss : 1.8022511005401611  | Training loss : 1.5986768007278442  |   Training accuracy: 0.5595499873161316 validation accuracy : 0.517372190952301
Number of epochs: 5 | Validation loss : 1.788769245147705  | Training loss : 1.4686111211776733  |   Training accuracy: 0.589775025844574 validation accuracy : 0.5239616632461548
Number of epochs: 6 | Validation loss : 1.8077868223190308  | Training loss : 1.3655238151550293  |   Training accuracy: 0.6147750020027161 validation accuracy : 0.5278554558753967
Number of epochs: 7 | Validation loss : 1.7925704717636108  | Training loss : 1.259329915046692  |   Training accuracy: 0.6416000127792358 validation accuracy : 0.5316493511199951
Number of epochs: 8 | Validation loss : 1.771490216255188  | Training loss : 1.1732298135757446  |   Training accuracy: 0.6649749875068665 validation accuracy : 0.5444288849830627
Number of epochs: 9 | Validation loss : 1.7830157279968262  | Training loss : 1.1098071336746216  |   Training accuracy: 0.6811000108718872 validation accuracy : 0.539536714553833
Number of epochs: 10 | Validation loss : 1.802015781402588  | Training loss : 1.0327491760253906  |   Training accuracy: 0.7012249827384949 validation accuracy : 0.5413338541984558
Number of epochs: 11 | Validation loss : 1.8209189176559448  | Training loss : 0.9608210921287537  |   Training accuracy: 0.7211250066757202 validation accuracy : 0.5444288849830627
Number of epochs: 12 | Validation loss : 1.8492668867111206  | Training loss : 0.8958105444908142  |   Training accuracy: 0.7372249960899353 validation accuracy : 0.546026349067688
Number of epochs: 13 | Validation loss : 1.8730000257492065  | Training loss : 0.8485578894615173  |   Training accuracy: 0.7502250075340271 validation accuracy : 0.5461261868476868
Number of epochs: 14 | Validation loss : 1.8870092630386353  | Training loss : 0.8059845566749573  |   Training accuracy: 0.7645750045776367 validation accuracy : 0.5495207905769348
CPU times: user 26min 15s, sys: 4.51 s, total: 26min 20s
Wall time: 26min 35s

#### Training and Validation Loss
![image](https://github.com/KAMAlhameedawi/cengcu-comparison-of-resnets/assets/149914341/f1e0947d-1afb-4087-995a-3e3fdb555455)

#### Training and Validation Accuracy

![image](https://github.com/KAMAlhameedawi/cengcu-comparison-of-resnets/assets/149914341/854b6a49-6946-4c69-8eef-e9f996f6d43f)





#### Comparison

Optionally, a comparison is made between the performance of the fine-tuned ResNet-50 model and a baseline model (e.g., a model trained from scratch on the same dataset) or other state-of-the-art approaches. The advantages of using transfer learning in this context are discussed.

### Test Accuracy for ResNet 50

Test Accuracy for ResNet 50 is :  0.547723650932312

# BASIC (40%)
## A Linear Classifier

## View Dataset for each batch

tensor([71, 17, 20, 62, 35, 77, 82, 32,  2, 30, 60, 22, 83, 39, 73, 72, 65, 67,
        88, 57, 49, 41, 74, 44, 26, 75, 87, 47, 52, 48,  0, 77])


![image](https://github.com/KAMAlhameedawi/cengcu-comparison-of-resnets/assets/149914341/2e0e1208-0dbb-4b4f-97ae-5b6d48ba1a0b)

#### Training

Number of epochs: 0 | Validation loss : 4.208994388580322  | Training loss : 4.356310844421387  |   Training accuracy: 0.06302499771118164 validation accuracy : 0.07358226925134659
Number of epochs: 1 | Validation loss : 4.211683750152588  | Training loss : 4.225301265716553  |   Training accuracy: 0.07244999706745148 validation accuracy : 0.07747603952884674
Number of epochs: 2 | Validation loss : 4.186000823974609  | Training loss : 4.201208114624023  |   Training accuracy: 0.07434999942779541 validation accuracy : 0.07408147305250168
Number of epochs: 3 | Validation loss : 4.208256721496582  | Training loss : 4.185935020446777  |   Training accuracy: 0.07612500339746475 validation accuracy : 0.08027156442403793
Number of epochs: 4 | Validation loss : 4.151335716247559  | Training loss : 4.1733293533325195  |   Training accuracy: 0.0765250027179718 validation accuracy : 0.07677716016769409
Number of epochs: 5 | Validation loss : 4.180080413818359  | Training loss : 4.167116641998291  |   Training accuracy: 0.07864999771118164 validation accuracy : 0.08077076822519302
Number of epochs: 6 | Validation loss : 4.1689934730529785  | Training loss : 4.166337966918945  |   Training accuracy: 0.07827500253915787 validation accuracy : 0.07967252284288406
Number of epochs: 7 | Validation loss : 4.1152424812316895  | Training loss : 4.1637091636657715  |   Training accuracy: 0.07750000059604645 validation accuracy : 0.08346645534038544
Number of epochs: 8 | Validation loss : 4.14443826675415  | Training loss : 4.155308723449707  |   Training accuracy: 0.07970000058412552 validation accuracy : 0.08176916837692261
Number of epochs: 9 | Validation loss : 4.1468119621276855  | Training loss : 4.152425289154053  |   Training accuracy: 0.07750000059604645 validation accuracy : 0.0790734812617302
Number of epochs: 10 | Validation loss : 4.143246650695801  | Training loss : 4.146578311920166  |   Training accuracy: 0.08327499777078629 validation accuracy : 0.07498003542423248
Number of epochs: 11 | Validation loss : 4.1534013748168945  | Training loss : 4.1468825340271  |   Training accuracy: 0.07970000058412552 validation accuracy : 0.07737620174884796
Number of epochs: 12 | Validation loss : 4.132030963897705  | Training loss : 4.14559268951416  |   Training accuracy: 0.08107499778270721 validation accuracy : 0.08107028901576996
Number of epochs: 13 | Validation loss : 4.135043621063232  | Training loss : 4.147431373596191  |   Training accuracy: 0.0791499987244606 validation accuracy : 0.08166933059692383
Number of epochs: 14 | Validation loss : 4.158107757568359  | Training loss : 4.143324851989746  |   Training accuracy: 0.07935000211000443 validation accuracy : 0.07667731493711472
Number of epochs: 15 | Validation loss : 4.13309383392334  | Training loss : 4.141639709472656  |   Training accuracy: 0.07864999771118164 validation accuracy : 0.08266773074865341
Number of epochs: 16 | Validation loss : 4.152688980102539  | Training loss : 4.13968563079834  |   Training accuracy: 0.08147499710321426 validation accuracy : 0.08186900615692139
Number of epochs: 17 | Validation loss : 4.128926753997803  | Training loss : 4.138754367828369  |   Training accuracy: 0.0801749974489212 validation accuracy : 0.08276756852865219
Number of epochs: 18 | Validation loss : 4.13217306137085  | Training loss : 4.132533073425293  |   Training accuracy: 0.08030000329017639 validation accuracy : 0.08186900615692139
Number of epochs: 19 | Validation loss : 4.128352165222168  | Training loss : 4.138171195983887  |   Training accuracy: 0.08004999905824661 validation accuracy : 0.08526358008384705

#### Training and Validation loss
![image](https://github.com/KAMAlhameedawi/cengcu-comparison-of-resnets/assets/149914341/c89ec2ce-4176-4510-958b-1f2622f459e1)

#### Training and Validation Accuracy

![image](https://github.com/KAMAlhameedawi/cengcu-comparison-of-resnets/assets/149914341/5c9f965e-34c8-4a34-abba-44ece072d4ca)

### Test Accuracy

Test Accuracy :  7.907348126173019  %



