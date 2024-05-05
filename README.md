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
Our pre-processing includes cropping the car images to include only the license plates. 
'get_inputs' returns these cropped images as a tensor, to be used in the LPRNet Model which makes use of non-descript sequence data. 
'get_segmented_images' returns a tensor of the segmented characters for each image, to be used in the Segmentation Model. This is done systematically through non-**** computer vision functions (cv2).
'get_labels' returns 
