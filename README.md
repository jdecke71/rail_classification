## Rail Classification

Binary classification of electrified rails.

#### Folder Structure

notebooks: Jupyter notebooks for collection and evaluation.

data: Contains folders for output images, railway shapefiles, and results of model evaluation.

resources: Text file for google maps api key - must be provided to use collections. GIS file for testing.

#### Project Outline

The project currently has three components:
* Image Collection
* Image Tagging
* Classification

##### Image Collection

Convert latitude/longitude points to satellite image for each railway. 

Each railway has an associated GIS shapefile containing the railway as a series of points. A selection of points is pulled and used to query the Google Static Maps api. The images are saved to file along with a csv with the coordinates as the file name. There is a notebook for each railway.

##### Image Tagging

A manual process of updating the csvs with a 1 if an image contains catenary lines or a 0 if the image does not.

##### Classification

Modeling and evaluation of image data.

The file 'ml_test.ipynb' is a script for loading images from file into a dataframe and completing a baseline evaluation of machine learning algorithms. Results are shown onscreen and saved to file. 

The first part of the script can be used to load the dataset then saved elsewhere to run in other algorithms.
