# CS153 Final Project
## Computer Vision in Papers, Please

## Requirements
### Download Tesseract 4.0
Download Tesseract 4.0 with "missing dlls" from [here](https://digi.bib.uni-mannheim.de/tesseract/). Then, within the code, change the location of tesseract in ```pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'```. Remember to add Tesseract to your PATH.

## Steps in Code (main.py)
<ol>
  <li>Import the image</li>
  <li>Crop the image</li>
  <li>Find the edge of the documents with Canny Edge Detection</li>
  <li>Resize the bounding box of the document edges</li>
  <li>Determining passport country with Color Detection</li>
  <li>Cropping and thresholding images of attributes from documents</li>
  <li>Using PyTesseract to read the cropped attributes</li>
  <li>Comparing attributes</li>
  <li>If all attributes have been correctly matched, the traveler has correct information. Else, the traveler has incorrect information</li>
</ol> 

## Files
### main.py
Main code
### BMMINI_Training
Files to train Tesseract on BMMini font
### MiniKylie_Training
Files to train Tesseract on MiniKylie fonr
### Final_Dataset
Contains all 69 screenshots used to test the code
### Seals
Contains images of correct seals for the entry permit
### QT-Box-Editor
Program used to correct Tesseract box files
### Final-Output.txt
Text file with all print statements and results from images in "Final_Dataset"
