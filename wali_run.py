from cv2 import cv2 
import csv
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def build_stats(data, new_file=True):
    if new_file: 
        cm = 'w'
        header = 'file_path, area_img, area_img_20perc, warn_found, area_warn, area_warn_perc, warn_in_top_half'
        data.insert(0, header)
    else: 
        cm = 'a'
        header = None
    data = [x.split(',') for x in data]
    with open('./output/results_summary.csv', cm, newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerows(data)

def get_warning_stats(root, file_path):
    # init vars
    h_warn = '-'
    w_warn = '-'
    warn_found = False
    warn_in_top_half = '-'
    area_warn = '-'
    area_warn_perc = '-'
    
    # Read image
    img = cv2.imread(root+file_path)
    h_img = img.shape[0]
    w_img = img.shape[1]
    
    # Basic image transformation
    image = img.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur
    blur = cv2.medianBlur(gray, 7)
    
    # Sharpen edges
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
    sharpen_blur = cv2.filter2D(blur, -1, sharpen_kernel)
    
    # Treshold color + Morphology to remove small noises
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thresh_1 = cv2.threshold(sharpen_blur, 5, 255, cv2.THRESH_BINARY_INV)[1]
    thresh_2 = cv2.threshold(sharpen_blur, 0, 250, cv2.THRESH_BINARY_INV)[1]
    thresh_3 = cv2.threshold(sharpen, 5, 255, cv2.THRESH_BINARY_INV)[1]
    thresh_4 = cv2.threshold(sharpen, 0, 250, cv2.THRESH_BINARY_INV)[1]
    thresh_5 = cv2.morphologyEx(thresh_1, cv2.MORPH_CLOSE, kernel, iterations=1)
    thresh_6 = cv2.morphologyEx(thresh_2, cv2.MORPH_CLOSE, kernel, iterations=1)
    thresh_7 = cv2.morphologyEx(thresh_3, cv2.MORPH_CLOSE, kernel, iterations=1)
    thresh_8 = cv2.morphologyEx(thresh_4, cv2.MORPH_CLOSE, kernel, iterations=1)

    for thresh in [thresh_1, thresh_2, thresh_3, thresh_4, thresh_5, thresh_6, thresh_7, thresh_8]:
    
        # prepare image version for the plots
        close = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        # Get contours
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        # Filter contours
        min_area = 10000 #(w_img/4)*(h_img/4)
        max_area = w_img*(h_img/2)
        max_contour = 0
        option_list = []

        for c in contours:
            area = cv2.contourArea(c)

            # 1st level filter only on area
            if area > min_area and area < max_area:         
                x, y ,w ,h = cv2.boundingRect(c)
                #cv2.drawContours(close, [c], -1, (0, 255, 0), 3) # GREEN
                smooth = 0.02
                approx = cv2.approxPolyDP(c, smooth*cv2.arcLength(c, True), True)

                # 2nd level filter on shape (rectangular)
                if len(approx) == 4:
                    #cv2.drawContours(close, [c], -1, (255, 0, 0), 4) # RED
                    ROI = image[y:y+h, x:x+w]
                    s = pytesseract.image_to_string(ROI).strip().lower()

                    # 3rd level filter on text 
                    if ('warning' in s): 
                        option_list.append([area, image, c, x, y ,w ,h])

        if option_list != []:     
            option_list = sorted(option_list, key=lambda x: x[0])
            choice = option_list[-1]  # largest area         
            area, image, c, x, y ,w ,h = choice    
            cv2.drawContours(image, [c], -1, (0, 0, 255), 3) # RED
            h_warn = h
            w_warn = w
            warn_found = True
            vert_pos_warn = y
            warn_in_top_half = (y+h_warn) < (h_img/2)
    
    # Get stats
    area_img = int(h_img*w_img)
    area_img_20perc = int(0.2*area_img)
    if not warn_found:
        cv2.imwrite('./output/imgs_no_warning/'+file_path, image)
    else:
        area_warn = int(h_warn*w_warn)
        area_warn_perc = round((area_warn*100)/area_img,1)
        cv2.imwrite('./output/imgs_warning_processed/'+file_path, image)
        
    res = f'{file_path}, {area_img}, {area_img_20perc}, {warn_found}, {area_warn}, {area_warn_perc}, {warn_in_top_half}'
    
    '''# resize image
    def resiz(img):
        width = int(img.shape[1] * 0.6)
        height = int(img.shape[0] * 0.6)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return img
    thresh = resiz(thresh)
    cv2.imshow('thresh', thresh)
    close = resiz(close)
    cv2.imshow('close_processed', close)
    image = resiz(image)
    cv2.imshow('image', image)
    cv2.waitKey()'''
    
    return res





if __name__ == "__main__":
    # Params
    root = './test/'
    new_file_res_file = True
    if not os.path.exists('./output/imgs_no_warning/'): os.makedirs('./output/imgs_no_warning/')
    if not os.path.exists('./output/imgs_warning_processed/'): os.makedirs('./output/imgs_warning_processed/')
    # run
    files = listdir(root)
    data = []
    for f in tqdm(files):
        res = get_warning_stats(root, str(f))
        data.append(res)
    build_stats(data, new_file=new_file_res_file)