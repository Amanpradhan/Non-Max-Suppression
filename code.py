import cv2
import random
import numpy as np

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
img = cv2.imread('example.png')
img = cv2.resize(img, (1000,1000))

cv2.rectangle(img,(0, 0), (800, 800),(0, 255, 0), 5)

def non_max_suppression_slow(boxes, overlapThresh):

	if len(boxes) == 0:
		return []

	pick = []
	
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]	

	area = (x2 - x1  + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]

		for pos in range(0, last):
			j = idxs[pos]

			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])

			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)

			overlap = float(w * h) / area[j]

			if overlap > overlapThresh:
				suppress.append(pos)
		idxs = np.delete(idxs, suppress)	
	return boxes[pick]		

l_rec = np.array([0, 0, 0, 0])
l_cir = np.array([ 0, 0, 0, 0])

def draw(zone_limit_x1, zone_limit_y1, zone_limit_x2, zone_limit_y2):
	# top left
	global l_cir, l_rec
	
	for _ in range(20):
		x1 = random.randint(zone_limit_x1, zone_limit_x2)
		y1 = random.randint(zone_limit_y1, zone_limit_y2)
		r = random.randint(16, 320)
		cv2.circle(img, (x1, y1), r, (255, 0, 0), 0)
		b = np.array([x1 - r, y1 - r, x1 + r, y1 + r])
		l_cir = np.vstack((l_cir, b))

		x2 = random.randint(zone_limit_x1, zone_limit_x2)
		y2 = random.randint(zone_limit_y1, zone_limit_y2)
		per_x = random.randint(16, 320)
		per_y = random.randint(16, 320)
		cv2.rectangle(img, (x2, y2), (x2+per_x, y2+per_y), (0, 255, 0), 2)
		b = np.array([x2, y2, x2+per_x, y2+per_y])
		l_rec = np.vstack((l_rec, b))

# top left zone
draw(0, 0, 400, 400)
# top right zone
draw(401, 0, 800, 400)
# bottom left zone
draw(0, 401, 400, 800)
# bottom right zone
draw(401, 401, 800, 800)
# centre zone
draw(200, 200, 600, 600)


images_rectangle = [
	("example.png",l_rec)
	]

# loop over the images
for (imagePath, boundingBoxes) in images_rectangle:
	# load the image and clone it
	print("[x] %d initial bounding boxes" % (len(boundingBoxes)))
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (1000,1000))
	orig = image.copy()

	# loop over the bounding boxes for each image and draw them
	for (startX, startY, endX, endY) in boundingBoxes:
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)

	# perform non-maximum suppression on the bounding boxes
	pick = non_max_suppression_slow(boundingBoxes, 0.1)
	print("[x] after applying non-maximum, %d bounding boxes" % (len(pick)))

	# loop over the picked bounding boxes and draw them
	for (startX, startY, endX, endY) in pick:
		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

	

images_circle = [
	("example.png",l_cir)	
	]
	
for (imagePath, boundingBoxes) in images_circle:
	# load the image and clone it
	print("[x] %d initial bounding boxes" % (len(boundingBoxes)))
	orig = image.copy()

	# loop over the bounding boxes for each image and draw them
	for (startX, startY, endX, endY) in boundingBoxes:
		startX = int(startX)
		startY = int(startY)
		endX = int(endX)
		endY = int(endY)
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)

	# perform non-maximum suppression on the bounding boxes
	pick = non_max_suppression_slow(boundingBoxes, 0.1)
	print("[x] after applying non-maximum, %d bounding boxes" % (len(pick)))

	# loop over the picked bounding boxes and draw them
	for (startX, startY, endX, endY) in pick:
		x = int((startX + endX) // 2)
		y = int((startY + endY) // 2)
		r = int((endY - startY) // 2)
		cv2.circle(image, ( x, y), r, (255,255,0), 3)

cv2.rectangle(image, (0, 0), (800, 800), (0, 255, 0), 5)

cv2.imshow('frame', image)
cv2.imshow('frame_orig', img)
cv2.waitKey(0)
cv2.destroyAllWindows()