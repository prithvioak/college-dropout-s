# college-dropout-s
Our project is on license plate recognition. Our code supports license plate preprocessing, character segmentation, and two different model implementations for character sequence recognition based on image input.

## To Run
Download the dataset here: https://www.inf.ufpr.br/vri/databases/tbFcZE-RodoSol-ALPR.zip (3 GB)
Create a folder in the project directory called 'data'
Navigate to the 'cars-br' folder in the dataset and add that to the 'data' folder.
Create a second folder called 'cars-br-cropped' in the data folder.
Run the preprocess function, which will populate the project with images, now cropped to only include the license plates.
Now, you can run both models on this data!

### Pre-process
Our pre-processing includes cropping the car images to include only the license plates. <br>
'get_inputs_LRPNet' returns these cropped images as a tensor, to be used in the LPRNet Model which makes use of non-descript sequence data. <br>
'get_labels_LPRNet' returns the labels of each image as a tensor in our integer to character mapping<br>
'get_segmented_images' returns a tensor of the segmented characters for each image, to be used in the Segmentation Model. This is done systematically through deterministic computer vision functions (cv2).<br>
'get_labels' returns the labels of each image as a tensor as one-hot encodings<br>

### LPRNetModel
Accuracy: 0 :(

### Segmented Model
Character-wise Accuracy: 44.4% <br>
Sequence-wise (True) Accuracy: 5.4%
