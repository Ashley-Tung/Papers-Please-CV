import re
import cv2
import numpy as np
import pytesseract
import math
import string
import glob
import argparse
import imutils
import sys
from datetime import datetime
from pytesseract import Output
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim

# Tutorial on tesseract:
# https://nanonets.com/blog/ocr-with-tesseract/

# ----------Global variables----------
# Set up tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Seals will always remain the same
entry_seal_1= cv2.imread("Seals/Entry_Seal_1.jpg")
entry_seal_1 = cv2.cvtColor(entry_seal_1, cv2.COLOR_BGR2RGB)
entry_seal_2= cv2.imread("Seals/Entry_Seal_2.jpg")
entry_seal_2 = cv2.cvtColor(entry_seal_2, cv2.COLOR_BGR2RGB)
# ------------------------------------


# Identify the shapes of our important documents
# Use Canny edge detection
def identify_shapes(img):

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(gray, 100, 200, 1)

    # cv2.imshow("canny",canny)

    cnts, hierarchy= cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Store areas of all contours
    all_areas= []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        all_areas.append(area)

    # Draw contours of largest and second-largest shapes
    sorted_contours = sorted(cnts, key=cv2.contourArea, reverse=True)
    largest_item = sorted_contours[0]
    
    cv2.drawContours(img, [largest_item], -1, (255, 0, 0), 10)

    second_largest_item = sorted_contours[1]
    cv2.drawContours(img, [second_largest_item], -1, (255, 0, 0), 10)
    plt.imshow(img)
    plt.show()

    # After we have identified the contours, we want the locations of the edges of the bounding box
    # So that we may know where the document properties are
    # relative to the edges of our document

    # Coordinates of bounding box of larger contour
    lsmallx, lsmally, llargex, llargey = math.inf, math.inf,0,0
    for coord in largest_item:
        if coord[0][0] < lsmallx:
            lsmallx = coord[0][0]
        if coord[0][1] < lsmally:
            lsmally = coord[0][1]
        if coord[0][0] > llargex:
            llargex = coord[0][0]
        if coord[0][1] > llargey:
            llargey = coord[0][1]

    # Coordinates of bounding box of second-largest contour
    ssmallx, ssmally, slargex, slargey = math.inf, math.inf, 0, 0
    for coord in second_largest_item:
        if coord[0][0] < ssmallx:
            ssmallx = coord[0][0]
        if coord[0][1] < ssmally:
            ssmally = coord[0][1]
        if coord[0][0] > slargex:
            slargex = coord[0][0]
        if coord[0][1] > slargey:
            slargey = coord[0][1]

    # Variables to make readability better
    largeCoordTop = [lsmallx,lsmally]
    largeCoordBot = [llargex,llargey]
    smallCoordTop = [ssmallx, ssmally]
    smallCoordBot = [slargex, slargey]

    return largeCoordTop, largeCoordBot, smallCoordTop, smallCoordBot

# Identify the country of origin based on color of passport edge
# Resource on color detection:
# https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
def detect_country(img):
    # RGB colors for passports are static and these values should all be mutually exclusive
    # No passport should be mistaken for another

    boundaries = [([57, 70, 58], [63, 74, 62]),
                  ([50, 77, 30], [52, 78, 37]),
                  ([95, 28, 10], [101, 34, 17]),
                  ([81, 32, 60], [87, 37, 66]),
                  ([129, 5, 19], [135, 12, 25]),
                  ([71, 40, 27], [79, 44, 34]),
                  ([38, 25, 81], [43, 30, 90])]

    countries = ["Arstotzka",
                 "Antegria",
                 "Impor",
                 "Kolechia",
                 "Orbristan",
                 "Republia",
                 "UF"]

    # Detect if the color exists
    countryInd = 0
    maxValue = 0
    country = ""
    currentBorder = ""
    
    for (lower, upper) in boundaries:
        # Create arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        # Find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(img, lower, upper)
        border = cv2.bitwise_and(img, img, mask=mask)

        # We want to check the color of the mask and find its corresponding country
        # We can sum the values in the mask, and the largest value mean
        # we found the most color within those boundaries
        colorSum = np.sum(border)
        if colorSum > maxValue:
            country = countries[countryInd]
            maxValue = colorSum
            currentBorder = border

        countryInd += 1

    plt.imshow(currentBorder)
    plt.show()

    return country

# Isolates components of entry permit
# This is univeral for all countries (EXCEPT ARSTOTZKA)
# Returns Name, passport number, date
def entry_permit_attr(entry_permit):
    # Get image of the components
    # plt.imshow(entry_permit)
    # plt.show()
    
    name = entry_permit[265:295, 43:394]
    pass_num = entry_permit[365:394, 43:394]
    reason = entry_permit[408:442, 200:394]
    date = entry_permit[506:535, 200:350]

    plt.imshow(name)
    plt.show()

    # Clean the components of noise with thresholding
    name = thresh_entry(name)
    pass_num = thresh_entry(pass_num)
    reason = thresh_entry(reason)
    date = thresh_entry(date)

    plt.imshow(name)
    plt.show()

    return name, pass_num, reason, date

# Clean up components of entry permit attributes for tesseract
# We will preprocess the image for better text recognition
# We will grayscale, noise removal, thresholding, dilation, eroding, opening
def thresh_entry(attribute_img):
    gray = cv2.cvtColor(attribute_img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)
    _, thresh = cv2.threshold(blur, 125, 200, cv2.THRESH_BINARY)

    # Morph open to remove noise and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    return opening

# Clean up components of passport attributes for tesseract
# We will preprocess the image for better text recognition
# We will grayscale, noise removal, thresholding, dilation, eroding, opening
def thresh_img(attribute_img, is_Orb):
    gray = cv2.cvtColor(attribute_img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)
    if not is_Orb:
        _, thresh = cv2.threshold(blur, 100, 200, cv2.THRESH_BINARY)
        # Morph open to remove noise and invert image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    else:
        _, thresh = cv2.threshold(blur, 170, 200, cv2.THRESH_BINARY)
        # Morph open to remove noise and invert image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        opening = 255 - opening

    return opening

# Isolates components of Antegria passport
# The components are Name, DOB, Issuing City, Date, Passport number
def ant_passport_attr(passport):
    name = passport[410:440, 16:300]
    dob = passport[292:325, 61:208]
    iss = passport[349:380, 61:240]
    date = passport[377:410, 61:208]
    pass_num = passport[437:475, 185:370]

    # Clean up components of noise with thresholding
    name = thresh_img(name, False)
    dob = thresh_img(dob, False)
    iss = thresh_img(iss, False)
    date = thresh_img(date, False)
    pass_num = thresh_img(pass_num, False)

    return name, dob, iss, date, pass_num

# Isolates components of Arstotzka passport
# The components are Name, DOB, Issuing City, Date, Passport number
def ar_passport_attr(passport):
    name = passport[255:290, 21:320]
    dob = passport[285:318, 204:380]
    iss = passport[338:365, 204:380]
    date = passport[363:395, 204:380]
    pass_num = passport[440:475, 22:200]

    # Clean up components of noise with thresholding
    name = thresh_img(name, False)
    dob = thresh_img(dob, False)
    iss = thresh_img(iss, False)
    date = thresh_img(date, False)
    pass_num = thresh_img(pass_num, False)

    return name, dob, iss, date, pass_num

# Isolate components of the Arstotzkan ID
# The components are District, Name, DOB
def ar_ID_attr(ID):
    district = ID[29:55, 14:334]
    name = ID[53:107, 149:365]
    dob = ID[112:142, 197:369]

    # Clean up components of noise with thresholding
    district = 255 -  thresh_img(district, False)
    name = thresh_img(name, False)
    dob = thresh_img(dob, False)

    return district, name, dob

# Isolates components of Impor passport
# The components are Name, DOB, Issuing City, Date, Passport number
def imp_passport_attr(passport):
    name = passport[250:285, 15:290]
    dob = passport[280:315, 198:355]
    iss = passport[335:363, 198:360]
    date = passport[360:400, 198:360]
    pass_num = passport[425:465, 180:361]

    # Clean up components of noise with thresholding
    name = thresh_img(name, False)
    dob = thresh_img(dob, False)
    iss = thresh_img(iss, False)
    date = thresh_img(date, False)
    pass_num = thresh_img(pass_num, False)

    return name, dob, iss, date, pass_num

# Isolates components of Kolechia passport
# The components are Name, DOB, Issuing City, Date, Passport number
def kol_passport_attr(passport):
    name = passport[281:320, 14:280]
    dob = passport[310:345, 192:370]
    iss = passport[365:393, 192:370]
    date = passport[390:420, 192:370]
    pass_num = passport[430:470, 160:370]

    # Clean up components of noise with thresholding
    name = thresh_img(name, False)
    dob = thresh_img(dob, False)
    iss = thresh_img(iss, False)
    date = thresh_img(date, False)
    pass_num = thresh_img(pass_num, False)

    return name, dob, iss, date, pass_num

# Isolates components of Orbristan passport
# The components are Name, DOB, Issuing City, Date, Passport number
def orb_passport_attr(passport):
    name = passport[280:320, 20:280]
    plt.imshow(name)
    plt.show()
    dob = passport[320:355, 70:230]
    iss = passport[375:405, 70:230]
    date = passport[400:430, 70:230]
    pass_num = passport[430:468, 25:230]

    # Clean up components of noise with thresholding
    # For Orbristan: Need to invert! Except for passport name
    name = thresh_img(name, False)
    dob = thresh_img(dob, True)
    iss = thresh_img(iss, True)
    date = thresh_img(date, True)
    pass_num = thresh_img(pass_num, True)

    plt.imshow(name)
    plt.show()


    return name, dob, iss, date, pass_num

# Isolates components of Republia passport
# The components are Name, DOB, Issuing City, Date, Passport number
def rep_passport_attr(passport):
    name = passport[250:285, 15:280]
    dob = passport[285:318, 65:240]
    iss = passport[336:365, 65:240]
    date = passport[360:390, 65:240]
    pass_num = passport[436:470, 166:375]

    # Clean up components of noise with thresholding
    name = thresh_img(name, False)
    dob = thresh_img(dob, False)
    iss = thresh_img(iss, False)
    date = thresh_img(date, False)
    pass_num = thresh_img(pass_num, False)

    return name, dob, iss, date, pass_num

# Isolates components of UF passport
# The components are Name, DOB, Issuing City, Date, Passport number
def uf_passport_attr(passport):
    name = passport[280:318, 15:280]
    dob = passport[310:340, 200:360]
    iss = passport[360:390, 195:360]
    date = passport[385:415, 195:360]
    pass_num = passport[430:470, 165:370]

    # Clean up components of noise with thresholding
    name = thresh_img(name, False)
    dob = thresh_img(dob, False)
    iss = thresh_img(iss, False)
    date = thresh_img(date, False)
    pass_num = thresh_img(pass_num, False)

    return name, dob, iss, date, pass_num

# Using tesseract to read text from passport
def read_passport(p_name, p_DOB, p_ISS, p_date, p_num):
    # Date information should not have spaces inside
    p_name_text = pytesseract.image_to_string(p_name, lang='eng2').strip()
    p_DOB_text = pytesseract.image_to_string(p_DOB, lang='eng2').replace(" ","").strip()
    p_ISS_text = pytesseract.image_to_string(p_ISS, lang='eng2').strip()
    p_date_text = pytesseract.image_to_string(p_date, lang='eng2').replace(" ","").strip()
    p_num_text = pytesseract.image_to_string(p_num, lang='eng2').strip()

    return p_name_text, p_DOB_text, p_ISS_text, p_date_text, p_num_text

# Use tesseract to read text from entry permit
def read_permit(e_name, e_num, e_reason, e_date):
    # Text in entry permit is all in caps
    e_name_text = pytesseract.image_to_string(e_name, lang='eng2').strip()
    e_num_text = pytesseract.image_to_string(e_num, lang='eng2').replace(" ","").strip()
    e_reason_text = pytesseract.image_to_string(e_reason, lang='eng2').strip()
    e_date_text = pytesseract.image_to_string(e_date, lang='eng2').replace(" ","").strip()

    return e_name_text, e_num_text, e_reason_text, e_date_text

# Use tesseract to read text from Arstotzkan ID
def read_ID(ID_distr, ID_name, ID_DOB):
    ID_distr_text = pytesseract.image_to_string(ID_distr, lang='eng3').strip()
    ID_name_text = pytesseract.image_to_string(ID_name, lang='eng3').replace("\n", " ").strip()
    ID_DOB_text = pytesseract.image_to_string(ID_DOB, lang='eng3').replace(" ","").strip()

    print(ID_name_text)

    return ID_distr_text, ID_name_text, ID_DOB_text


# Determine within a percentage how much of a match two texts are
# Percentage is in decimal format
def determine_match(text_1, text_2, percentage):
    if len(text_1) != len(text_2):
        return False
    else:
        matches = 0
        for i in range(len(text_1)):
            if text_1[i] == text_2[i]:
                matches += 1
        if (matches/len(text_1)) >= percentage:
            return True
        else:
            return False

# Compare text from entry permit and passport 
def compare_passport_entry(passport_info, permit_info):
    # passport_info: [p_name_text, p_DOB_text, p_ISS_text, p_date_text, p_num_text]
    # permit_info: [e_name_text, e_num_text, e_reason_text, e_date_text]

    # Compare names
    # Valid: match
    # Format the names: passport is Last, First; entry permit is FIRST LAST 
    # to first last
    passport_name = (' '.join(passport_info[0].split(",")[::-1])).strip().lower()
    permit_name = permit_info[0].lower()
    name_bool = determine_match(passport_name, permit_name, 0.95)

    # Compare passport number
    # Valid: match
    number_bool = determine_match(passport_info[4], permit_info[1], 0.95)

    # Determine if date is valid
    # Valid: after 1983.01.01
    # Since the only characters allowed are numbers and period, all other punctuation points and letters should be removed
    passport_info_cleaned = ''.join(i for i in passport_info[3] if i in "1234567890.") 
    permit_info_cleaned = ''.join(i for i in permit_info[3] if i in "1234567890.") 

    passport_date = datetime.strptime(passport_info_cleaned,'%Y.%m.%d')
    permit_date = datetime.strptime(permit_info_cleaned,'%Y.%m.%d')
    current_day = datetime(1983, 1, 1)
    date_bool = (passport_date > current_day) & (permit_date > current_day)

    return (name_bool & number_bool & date_bool)

# Compare text from Passport and ID for Arstotzka
def compare_passport_ID(passport_info, ID_info):
    # passport_info: [p_name_text, p_DOB_text, p_ISS_text, p_date_text, p_num_text]
    # ID info: [ID_distr_text, ID_name_text, ID_DOB_text]

    # Determine if District is correct
    # Remove the "DISTRICT" from the end of the string
    district_bool = district(ID_info[0][:-9])

    # Determine if names match
    # Format of passport is Last, First; format of ID is LAST, FIRST
    # Remove white spaces as well, as sometimes tesseract adds an extra whitespace
    passport_name = passport_info[0].lower().replace(" ", "")
    ID_name = ID_info[1].lower().replace(" ", "")
    name_bool = determine_match(passport_name, ID_name, 0.95)

    # Determine if DOB match
    # Since the only characters allowed are numbers and period, all other punctuation points and letters should be removed
    passport_DOB_cleaned = ''.join(i for i in passport_info[1] if i in "1234567890.") 
    ID_DOB_cleaned = ''.join(i for i in ID_info[2] if i in "1234567890.") 

    passport_DOB = datetime.strptime(passport_DOB_cleaned,'%Y.%m.%d')
    ID_DOB = datetime.strptime(ID_DOB_cleaned,'%Y.%m.%d')

    DOB_bool = determine_match(str(passport_DOB), str(ID_DOB), 0.90)

    return (district_bool & name_bool & DOB_bool)

# Determine if the stamp on the entry permit is valid
def permit_seal(entry_permit):
    # Stamp will always be above a certain height on the entry permit
    cropped = entry_permit[:200,:]

    plt.imshow(entry_permit)
    plt.show()

    plt.imshow(cropped)
    plt.show()

    # Use color ranges to isolate entry permit
    color1 = np.asarray([155, 39, 35])   
    color2 = np.asarray([250, 137, 128])   
    mask = cv2.inRange(cropped, color1, color2)

    # Dilate the mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations = 2)

    plt.imshow(mask)
    plt.show()

    # Find bounding box of mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if cnts == ():
        # No Seal found
        return False
    else:
        c = max(cnts, key=cv2.contourArea)

    # draw the contours of c
    # cv2.drawContours(cropped, [c], -1, (0, 0, 255), 2)
    lsmallx, lsmally, llargex, llargey = math.inf, math.inf,0,0
    for coord in c:
        if coord[0][0] < lsmallx:
            lsmallx = coord[0][0]
        if coord[0][1] < lsmally:
            lsmally = coord[0][1]
        if coord[0][0] > llargex:
            llargex = coord[0][0]
        if coord[0][1] > llargey:
            llargey = coord[0][1]

    # show the output image
    # cv2.rectangle(cropped, (lsmallx, lsmally), (llargex, llargey), (0, 255, 0), 2)
    # print(lsmallx, lsmally, llargex, llargey)
    # print(llargex-lsmallx, llargey-lsmally)

    # Mask the image
    masked = cv2.bitwise_and(cropped, cropped, mask = mask)

    # We want to determine how similar the seals are to one another
    # using Structural Similarity Index (SSIM)
    # Resource: https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
    # Since there will be differences due to masking and text on the seal,
    # the difference should be within a certain range
    
    # First, default icons are 70 x 70
    # The seal in game will always be bigger so we havbe to reshape the seal
    resized_seal_1 = cv2.resize(entry_seal_1, (llargex-lsmallx, llargey-lsmally), interpolation = cv2.INTER_NEAREST)
    resized_seal_2 = cv2.resize(entry_seal_2, (llargex-lsmallx, llargey-lsmally), interpolation = cv2.INTER_NEAREST)
    # Extract seal from cropped image and compare it to the original seal
    extracted_seal = masked[lsmally:llargey, lsmallx:llargex]

    extracted_seal_mask = cv2.inRange(extracted_seal, color1, color2)

    # Dilate the mask
    kernel = np.ones((3,3), np.uint8)
    extracted_seal_mask = cv2.erode(extracted_seal_mask, kernel, iterations = 1)
    extracted_seal_mask = cv2.dilate(extracted_seal_mask, kernel, iterations = 1)

    # ----------------------------------
    resized_seal_1_mask = cv2.inRange(resized_seal_1, color1, color2)

    # Dilate the mask
    kernel = np.ones((3,3), np.uint8)
    resized_seal_1_mask = cv2.erode(resized_seal_1_mask, kernel, iterations = 1)
    resized_seal_1_mask = cv2.dilate(resized_seal_1_mask, kernel, iterations = 1)

    # -----------------------------------
    resized_seal_2_mask = cv2.inRange(resized_seal_2, color1, color2)

    # Dilate the mask
    kernel = np.ones((3,3), np.uint8)
    resized_seal_2_mask = cv2.erode(resized_seal_2_mask, kernel, iterations = 1)
    resized_seal_2_mask = cv2.dilate(resized_seal_2_mask, kernel, iterations = 1)

    # -----------------------------------

    (score_1, diff_1) = compare_ssim(extracted_seal_mask, resized_seal_1_mask, full=True, multichannel=True)
    (score_2, diff_1) = compare_ssim(extracted_seal_mask, resized_seal_2_mask, full=True, multichannel=True)

    plt.imshow(extracted_seal_mask)
    plt.show()
    plt.imshow(resized_seal_2_mask)
    plt.show()

    print("Score to entry seal 1: ", score_1)
    print("Score to entry seal 2: ", score_2)

    # Score needs to be above 0.85
    seal_bool = (score_1 > 0.85) or (score_2 > 0.85)
    return seal_bool

# Compare Issuing City text to hard-coded text
def issuing_city(country, p_ISS_text):
    percentage = 0.95
    if country == "Arstotzka":
        match = (determine_match(p_ISS_text, "Orvech Vonor", percentage) or
                determine_match(p_ISS_text, "East Grestin", percentage) or
                determine_match(p_ISS_text, "Paradizna", percentage))
    elif country == "Antegria":
        match = (determine_match(p_ISS_text, "St. Marmero", percentage) or
                determine_match(p_ISS_text, "Glorian", percentage) or
                determine_match(p_ISS_text, "Outer Grouse", percentage))
    elif country == "Impor":
        match = (determine_match(p_ISS_text, "Enkyo", percentage) or
                determine_match(p_ISS_text, "Haihan", percentage) or
                determine_match(p_ISS_text, "Tsunkeido", percentage))
    elif country == "Kolechia":
        match = (determine_match(p_ISS_text, "Yurko City", percentage) or
                determine_match(p_ISS_text, "Vedor", percentage) or
                determine_match(p_ISS_text, "West Grestin", percentage))
    elif country == "Orbristan":
        match = (determine_match(p_ISS_text, "Skal", percentage) or
                determine_match(p_ISS_text, "Lorndaz", percentage) or
                determine_match(p_ISS_text, "Mergerous", percentage))
    elif country == "Republia":
        match = (determine_match(p_ISS_text, "True Glorian", percentage) or
                determine_match(p_ISS_text, "Lesrenadi", percentage) or
                determine_match(p_ISS_text, "Bostan", percentage))
    else: # UF
        match = (determine_match(p_ISS_text, "Great Rapid", percentage) or
                determine_match(p_ISS_text, "Shingleton", percentage) or
                determine_match(p_ISS_text, "Korista City", percentage))

    return match

# Compare District of Arstotzkan ID to hard-coded text
def district(district):
    percentage = 0.95
    match = (determine_match(district, "ATLAN", percentage) or
            determine_match(district, "VESCILLO", percentage) or
            determine_match(district, "BURNTON", percentage) or
            determine_match(district, "OCTOVALIS", percentage) or
            determine_match(district, "GENNISTORA", percentage) or
            determine_match(district, "LENDIFORMA", percentage) or
            determine_match(district, "WOZENFIELD", percentage) or
            determine_match(district, "FARDESTO", percentage))
    return match

# Resize documents
def resize(top, bot, width, height):
    # Make sure the dimensions are to what we want
    if (bot[0] - top[0] < width):
        temp = (width - (bot[0] - top[0]))//2
        top[0] -= temp
        bot[0] += temp
    if (bot[1] - top[1] < height):
        temp = (height - (bot[1] - top[1]))//2
        top[1] -= temp
        bot[1] += temp

    return top, bot

# Runs all the needed functions in order to compare
def compare(img_name):
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.show()

    # We only want to look at the table in the game
    # Don't forget, array is in (y,x) format
    img = img[345:1040, 635:1850]

    plt.imshow(img)
    plt.show()

    # Coordinates of bounding box of items
    # We are only looking at two items per country
    # For Arstotzka, passport and ID
    # For all other countries, passport and entry permit
    largeCoordTop, largeCoordBot, smallCoordTop, smallCoordBot = identify_shapes(img)

    # Determine the country of origin
    # We are looking for a specific shade for the passport
    country = detect_country(img)
    print("Country: ", country)

    # If the country is Arstotzka, then we are only looking at the passport
    # Given by the bounding box of largeCoordTop, largeCoordBot
    if country == "Arstotzka":

        # We want 390 by 485 for the passport
        # Resize passport
        largeCoordTop, largeCoordBot = resize(largeCoordTop, largeCoordBot, 390, 285)

        # We want 390 by 110 for the ID
        # Resize ID
        smallCoordTop, smallCoordBot = resize(smallCoordTop, smallCoordBot, 390, 210)

        # Isolate the passport and ID according to its bounding box
        passport = img[largeCoordTop[1]:largeCoordBot[1],largeCoordTop[0]:largeCoordBot[0]]
        ID = img[smallCoordTop[1]:smallCoordBot[1],smallCoordTop[0]:smallCoordBot[0]]

        # We will get the image of components of the Arstotzka passport
        # Name, DOB, Issuing City, Date, Passport number
        p_name, p_DOB, p_ISS, p_date, p_num = ar_passport_attr(passport)

        # We will get the image components of the Arstotzka ID
        # District, name, DOB
        ID_distr, ID_name, ID_DOB = ar_ID_attr(ID)

        # Identify the text within the passport
        p_name_text, p_DOB_text, p_ISS_text, p_date_text, p_num_text = read_passport(p_name, p_DOB, p_ISS, p_date, p_num)

        # Identify the text within the ID
        ID_distr_text, ID_name_text, ID_DOB_text = read_ID(ID_distr, ID_name, ID_DOB)

        # Compare text in passport and ID
        passport_info = [p_name_text, p_DOB_text, p_ISS_text, p_date_text, p_num_text]
        ID_info = [ID_distr_text, ID_name_text, ID_DOB_text]

        passport_ID_match = compare_passport_ID(passport_info, ID_info)

        # Compare Issuing City
        issuing_city_match = issuing_city(country, p_ISS_text)

        print("Name: ", p_name_text)
        print("ID Name: ", ID_name_text)
        print("DOB: ", p_DOB_text)
        print("ID DOB: ", ID_DOB_text)
        print("ISS: ", p_ISS_text)
        print("ID District: ", ID_distr_text)
        print("Date: ", p_date_text)
        print("Num: ", p_num_text)

        return passport_ID_match & issuing_city_match

    # If the country is not Arstotzka
    # We are looking at passport and entry permit
    else:
        # We want 390 by 485 for the passport
        # Resize passport
        smallCoordTop, smallCoordBot = resize(smallCoordTop, smallCoordBot, 390, 485)

        # We want 450 by 600 for the Entry Permit
        # Resize entry permit
        largeCoordTop, largeCoordBot = resize(largeCoordTop, largeCoordBot, 450, 600)

        # Isolate the entry permit, which will be the largest element for countries NOT Arstotzka
        # Isolate the passport according to its bounding box
        entry_permit = img[largeCoordTop[1]:largeCoordBot[1],largeCoordTop[0]:largeCoordBot[0]]
        passport = img[smallCoordTop[1]:smallCoordBot[1],smallCoordTop[0]:smallCoordBot[0]]

        # plt.imshow(entry_permit)
        # plt.show()

        # We will get the image of the components of the entry permits
        # Name, passport number, date
        e_name, e_num, e_reason, e_date = entry_permit_attr(entry_permit)

        # As well as components of the passports
        # Name, DOB, Issuing City, Date, Passport number
        if country == "Antegria":
            p_name, p_DOB, p_ISS, p_date, p_num = ant_passport_attr(passport)
        elif country == "Impor":
            p_name, p_DOB, p_ISS, p_date, p_num = imp_passport_attr(passport)
        elif country == "Kolechia":
            p_name, p_DOB, p_ISS, p_date, p_num = kol_passport_attr(passport)
        elif country == "Orbristan":
            p_name, p_DOB, p_ISS, p_date, p_num = orb_passport_attr(passport)
        elif country == "Republia":
            p_name, p_DOB, p_ISS, p_date, p_num = rep_passport_attr(passport)
        else:
            p_name, p_DOB, p_ISS, p_date, p_num = uf_passport_attr(passport)

        # Identify the text within the passport
        p_name_text, p_DOB_text, p_ISS_text, p_date_text, p_num_text = read_passport(p_name, p_DOB, p_ISS, p_date, p_num)
        e_name_text, e_num_text, e_reason_text, e_date_text = read_permit(e_name, e_num, e_reason, e_date)

        print("Name: ", p_name_text)
        print("Entry Name: ", e_name_text)
        print("DOB: ", p_DOB_text)
        print("ISS: ", p_ISS_text)
        print("Date: ", p_date_text)
        print("Entry Date: ", e_date_text)
        print("Num: ", p_num_text)
        print("Entry Num: ", e_num_text)
        print("Entry Reason: ", e_reason_text)
        

        passport_info = [p_name_text, p_DOB_text, p_ISS_text, p_date_text, p_num_text]
        permit_info = [e_name_text, e_num_text, e_reason_text, e_date_text]

        passport_permit_match = compare_passport_entry(passport_info, permit_info)

        # Compare Seal 
        # Looking at seal on entry permit
        permit_seal_match = permit_seal(entry_permit)

        # Compare Issuing City
        issuing_city_match = issuing_city(country, p_ISS_text)

        print("Passport - Permit match:", passport_permit_match)
        print("ISS match:", issuing_city_match)
        print("Permit Seal match:", permit_seal_match)

    
        return passport_permit_match & issuing_city_match & permit_seal_match
    

def main():
    dir = 'Final_Dataset/'
    img_name = 'Obristan_Name.jpg'
    print("Image name: ", img_name)
    print("Final: ", compare(dir + img_name))

   # Change folder for testing
    # dir = glob.glob("Final_Dataset/*.jpg")
    # false = 0
    # true = 0

    # Output into text file
    # sys.stdout = open("Final_Output.txt", "wt")

    # # In the Screenshots_pssprt_EntryPermit folder, we have 14 correct and 26 incorrect
    # for img in dir:
    #     print("Image name: ", img)
    #     result = compare(img)
    #     print("Final:", result)
    #     print("\n")
    #     if result:
    #         true += 1
    #     else:
    #         false += 1
    
    # print("Final Results. True: ", true, " False: ", false)

    # sys.stdout.close()

    # Final Dataset: Real True: 31, Real False: 38

if __name__ == "__main__":
    main()