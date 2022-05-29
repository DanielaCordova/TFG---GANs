# StyleGAN

When looking at an image, one unconsciously takes apart the content of the image from its style. For example, both Da Vinci's Mona Lisa and Monet's Woman with a Parasol share a woman in their content, but clearly, those women are not painted in the same style. Another example of this concept is that, although two people may wear the same t-shirt, they would probably do it in different styles. 

Using these examples as a reference, a style can be defined in the context of this work as a variation of any feature in an image. Relating the definition with examples, in the paintings, the women's hair can vary its style being long, short, curly, or straight, while the t-shirts present variations in their style by changing their color or the length of their sleeves.

The StyleGAN is named after this concept because not only it can generate highly realistic images, but because it is capable of changing and mixing styles thanks to two different techniques: random noise injection and style mixing. A deeper insight into this architecture's capability to alter and work with styles will be presented further in this work.

To handle styles, StyleGAN's Generator takes a random noise vector and a constant vector as an input. The idea behind having two different inputs is that the constant vector will travel through several convolutional layers and end up as the generated image, whereas the random noise vector is processed by some fully connected layers to generate other that represents the styles that will be applied to the random noise vector at various points during the generation process. The StyleGAN Discriminator, on the other hand, is quite similar to the Conditional GAN Discriminator, but it also applied some changes for it to keep up with the Generator. A whole intuition of the complete architecture is given here:

![StyleGanScheme](https://user-images.githubusercontent.com/60478676/170843665-e9574380-f6ba-4839-9ec9-2e25d61d94e4.svg)


# Code Structure

## Classes

- StyleDiscriminator.py : Contains the StyleGAN Discriminators class and that of the Blocks that it's composed of
- StyleGenerator.py : Contains the StyleGAN Generator, as well as all its parts
- TrainingStyleGan.py: Contains the Trainer class that is used for the StyleGAN


## Executables

- generateSamples.py : Generates a given number of samples from either a previously trained model that is given, or a new one from scratch without training the models
- trainingStyle.py : Trains a model, that is either previously trained or not, for the number of epochs given for each step of the progressive growing
- generateStyleMixing.py : Mixes the styles of two given images and saves them. This is done with a previously trained model. 





