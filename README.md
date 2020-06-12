## Face Recognition with Python & TensorFlow

### How to run the application

#### Sample data collection

```> python sampler.py```

Execute the "sampler.py", this will prompt for and input, where you enter your name. After which the webcam will be opened and started detecting faces from the webcam frame. Once a face is detected, it will be saved in the /sample/[name] directory. where 'name' is the name you entered in the initial prompt. A set of 10 images will be saved as [i].jpg where i ranges from 0 to 9

Prior to saving the images, the images will go through the following step.

- convert to grayscale
- find the face(s) in the image
- cut the face area based on the bouding box
- remove a 20% margin around the face area, this is to remove the possible noise
- histogram equialization to normalize the pixel intensity
- resize the image to 50 x 50 

If a face is detected, the output window will show the red colored bouding box around the detected faces(s)



#### Training a model

```> python trainer.py```

Executing the "trainer.py" will read all the directories (one directory per each sample person) in the sample directory while looping through all the images in each directory and saving the pixel arrays in a variable called "samples". At the sametime a "label" array si also created based on the sample directory indexes.

after the model is trained, it will be saved in the /models directory.


#### Testing

```> python tester.py```

Run the "tester.py" script to test the application. This will open the webcam and detect the face(s) from the input frame and do a prediction of the detected face(s) using the trained model.

The output frame will show the label of the predicted face based on the highest probability.







