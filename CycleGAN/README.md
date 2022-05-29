# CycleGAN

A CycleGAN (https://arxiv.org/abs/1703.10593) is essentially two modules facing each other, where each one learns a mapping function from one class to another. If the CycleGAN is set to transform images from class A to class B and vice-versa, it receives a sample from class A, and first runs that input through the first module which produces a fake sample of class B. Then, that fake sample goes through the second module, producing a new fake sample of class A. Each module in the CycleGAN is in the end a GAN, so it contains a Generator and a Discriminator, making up to 2 Generators and 2 Discriminators contained in one CycleGAN.

The CycleGAN learns by comparing the final sample of class A to the original one because, in the end, if the final sample resembles the original one, that means that both the transformation from class A to class B and the one from class B to class A have worked correctly. Furthermore, assuming the CycleGAN has been appropriately trained, the two components can be separated to produce two independent models capable of transferring features from one class to a sample of another class.

By using unsupervised learning, CycleGAN can learn a mapping function from one image domain to another. This implies that the dataset doesn't need to contain samples of a class transformed into the other. Instead of learning the features of some transformed samples, CycleGAN learns to perform this transformation by telling Generator Networks to learn a mapping from domain X to what seems to be a picture from domain Y and vice-versa.

So, the mapping functions between two domains X and Y learned by the model would be.

     G : X → Y
     F : Y → X

Given training samples x ε X and y ε Y and data distributions denoted by x data(x) and y data(y), the architecture consists of two adversarial discriminators DX and DY , where DX aims to distinguish between images x and translated images F(y); in the same way, DY aims to discriminate between y and G(x).

In addition, two new loss functions are introduced. The first one, adversarial loss, is used to match the distribution of generated images to the distribution of data in the target domain, while the second one, cycle consistency losses, keeps the learned mappings G and F from contradicting one another. Both mapping functions are subjected to adversarial losses.

![Complete-Structure](https://user-images.githubusercontent.com/60478676/170842020-ea1f8756-a752-40de-a6b4-1aafca54866e.jpg)

In  this figure it can be seen the complete structure of a CycleGAN that transforms horses into zebras and vice versa. The first generator transforms a horse into a zebra, and then this generated image is passed on to the other generator to make a horse. 

# Code Structure

## Folders

- [PreprocessDatasets](PreprocessDatasets): Contains the preprocess datasets used for training and generating samples
- [models](models): Contains pre-trained models that can be used for generating samples
- [CycleTraining](CycleTraining) : Folder in which generated and real images, as well as the loss function grafic will be saved 


## Classes

- [CycleDiscriminator.py](CycleDiscriminator.py): Contains the CycleGANs Discriminator class
- [CycleGenerator.py](CycleGenerator.py): Contains the CycleGANs Generator


## Executables

- [generateSamples.py](generateSamples.py): Generates a given number of samples from either a previously trained model that is given, or a new one from scratch without training the models
- [trainingCycle.py](TrainSimpleGan.py): Trains a model, that is either previously trained or not, for the number of epochs give
