# Definition of a Conditional GAN

Conditional GANs are generative adversarial networks that contain additional information on the input to condition the Generator and Discriminator during training. This auxiliary data might theoretically be anything, such as a class classification, a set of tags, or even a written explanation. In the context of this work, this additional information will be a label indicating one of the multiple classes. 

The fact that the Conditional Generator's input is extended with the label of the sample to generate, implies that both the Conditional Generator and the Conditional Discriminator need to learn a way to map the features of the samples in each class to their label. Then, the Conditional Generator learns this in order to provide control to the user over the features of the generated sample. Furthermore, the Conditional Discriminator now needs to detect images paired with the wrong label regardless of whether the image is real or fake, and fake images paired with the correct label.

As a result, producing realistic-looking data via the Conditional GAN Generator is insufficient to fool the Discriminator. It's also important that the examples it generates match their labels. After the Generator has been fully trained, the Conditional GAN can synthesize any sample from any class by feeding it the desired label. To clarify the idea that this architecture introduces, the next image shows a sketch of how everything is put together.

![CGAN_ALL](https://user-images.githubusercontent.com/60478676/170842272-21b30782-79a4-4008-a711-d25e32fd1ce0.png)
