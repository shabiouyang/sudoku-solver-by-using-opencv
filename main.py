import cv2
import numpy as np
import math

#solove sudoku
def findNextCellToFill(grid, i, j):
    for x in range(i,9):
        for y in range(j,9):
            if grid[x][y] == 0:
                return x,y
    for x in range(0,9):
        for y in range(0,9):
            if grid[x][y] == 0:
                return x,y
    return -1,-1

def isValid(grid, i, j, e):
    rowOk = all([e != grid[i][x] for x in range(9)])
    if rowOk:
        columnOk = all([e != grid[x][j] for x in range(9)])
        if columnOk:
            # finding the top left x,y co-ordinates of the section containing the i,j cell
            secTopX, secTopY = 3 *int(i/3), 3 *int(j/3)
            for x in range(secTopX, secTopX+3):
                for y in range(secTopY, secTopY+3):
                    if grid[x][y] == e:
                        return False
                return True
    return False

def solveSudoku(grid, i=0, j=0):
    i,j = findNextCellToFill(grid, i, j)
    if i == -1:
        return True
    for e in range(1,10):
        if isValid(grid,i,j,e):
            grid[i][j] = e
            if solveSudoku(grid, i, j):
                return True
            # Undo the current cell for backtracking
            grid[i][j] = 0
    return False

# get biggestblob
def orderPoints(approx):
	biggestBlob = []
	biggestBlob.append(approx[0][0])
	biggestBlob.append(approx[3][0])
	biggestBlob.append(approx[2][0])
	biggestBlob.append(approx[1][0])
	print(biggestBlob)
	return biggestBlob
def transformAndResize(image, biggestBlob):
	transformed = fourpointtransform(image, biggestBlob)
	transformedImg = transformed
	out = cv2.resize(transformed, (360, 375), 0, 0, cv2.INTER_CUBIC)
	out = out[0:360, 0:360]
	return out
def fourpointtransform(image, orderedPoints):
	tl = orderedPoints[0]
	tr = orderedPoints[1]
	br = orderedPoints[2]
	bl = orderedPoints[3]
	max_width = int (max(
                math.sqrt(pow((br[0] - bl[0]), 2) + pow((br[1] - bl[1]), 2)),
                math.sqrt(pow((tr[0] - tl[0]), 2) + pow((tr[1] - tl[1]), 2))))
	max_height = int (max(
                math.sqrt(pow((tr[0] - br[0]), 2) + pow((tr[1] - br[1]), 2)),
                math.sqrt(pow((tl[0] - bl[0]), 2) + pow((tl[1] - bl[1]), 2))))
	inputQuad = []
	inputQuad.append([tl[0], tl[1]])
	inputQuad.append([tr[0], tr[1]])
	inputQuad.append([br[0], br[1]])
	inputQuad.append([bl[0], bl[1]])
	outputQuad = []
	outputQuad.append([0, 0])
	outputQuad.append([max_width, 0])
	outputQuad.append([max_width, max_height])
	outputQuad.append([0, max_height])
	inputQuad = np.float32(inputQuad)
	print(inputQuad)
	outputQuad = np.float32(outputQuad)
	matrix = cv2.getPerspectiveTransform(inputQuad, outputQuad)
	warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
	rows,cols,channel = warped.shape
	rot = cv2.getRotationMatrix2D((rows/2,cols/2),-90,1)
	dst = cv2.warpAffine(warped,rot,(cols,rows))
	return dst
# main function
image = cv2.imread('1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 1.5, 1.5)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
#canny = cv2.Canny(thresh, 100, 200, 3)
#cv2.imshow('shabi',canny)
#cv2.waitKey(0)
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
maxArea = 0.0
contoursindex = 0
for c in range(len(contours)):
	area = cv2.contourArea(contours[c])
	if area > 10000:
		perimeter = cv2.arcLength(contours[c], True)
		approx = cv2.approxPolyDP(contours[c], 0.02*perimeter, True)
		#print(approx[1])
		if (area > maxArea and len(approx) == 4):
			maxArea = area
			biggestBlob = orderPoints(approx)
			result = transformAndResize(image.copy(), biggestBlob)
#cv2.imshow('shabi', result)
#cv2.waitKey(0)
######################################################
## segment picture
######################################################

for i in range(9):
	for j in  range(9):
		getsubimg = result[(i*40):((i+1)*40), (j*40):((j+1)*40)]
		cv2.imwrite('./numbers/%d.jpg' %((9*i)+j), getsubimg)

samples = []
for c in range(81):
    img  = cv2.imread('./numbers/%d.jpg' %c)       
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)    
    #_, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    normalized_roi = thresh/255
    sample = normalized_roi.reshape((1,1600))
    samples.append(sample)
np.save('samples.npy',samples)
samples = np.load('samples.npy')
number = np.array(samples, np.float32)
number = number.reshape((81,1600))

train_label = [5,3,0,0,7,0,0,0,0,
               6,0,0,1,9,5,0,0,0,
               0,9,8,0,0,0,0,6,0,
               8,0,0,0,6,0,0,0,3,
               4,0,0,8,0,3,0,0,1,
               7,0,0,0,2,0,0,0,6,
               0,6,0,0,0,0,2,8,0,
               0,0,0,4,1,9,0,0,5,
               0,0,0,0,8,0,0,7,9]
k = 81             
train_data = number[:k]  
train_label = np.array(train_label, np.float32)
train_label = train_label.reshape((train_label.size,1))
train_label = train_label[:k]
######################################################################
#SVM
'''svm = cv2.ml.SVM_create() 
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(1.0)
string = svm.train(train_data, cv2.ml.ROW_SAMPLE, train_label)'''
########################################################################
#kNN
model = cv2.ml.KNearest_create()
model.train(train_data,cv2.ml.ROW_SAMPLE,train_label)
retval, results, neigh_resp, dists = model.findNearest(number, 1)
string = results.ravel()
#print(train_label.reshape(1,len(train_label))[0])
#print(string)
#######################################################################
#ANN
#ann = cv2.ml.ANN_MLP_create()  
#ann.setLayerSizes(np.array([2, 10, 10, 1]))  # 必须首先执行此行  
#ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)  
#ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)  
#ann.setBackpropWeightScale(0.1)  
#ann.setBackpropMomentumScale(0.1)  
#ret = ann.train(train_data, cv2.ml.ROW_SAMPLE, train_label)  
#string, res = ann.predict(number) 
#####################################################################
soduko = string.reshape(9, 9)
#print(soduko)
solveSudoku(soduko)
c = []
for i in range(9):
	for j in range(9):
		n = int(soduko[i][j])
		c.append(n)
#print(soduko[0][0])
c = np.array(c, dtype = int)
s = c.reshape(9,9)
image = cv2.resize(image, (540, 960), 0, 0, cv2.INTER_CUBIC)
for i in range(9):
    for j in range(9):
        x = int(((i+0.25)*55)+15)
        y = int(((j+0.5)*55)+225)
        cv2.putText(image,str(s[j][i]),(x,y), 3, 1.4, (0, 0, 255), 2, cv2.LINE_AA)
#print(number_boxes)
#cv2.namedWindow("img", cv2.WINDOW_NORMAL);
cv2.imshow("img", image)
cv2.waitKey(0)