import numpy as np
from matplotlib import pyplot as plt
import cv2

def generatePloygon(n):
    return np.random.randint(0,400,(n,2))

def plotPolygon(points,block):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(points[:, 0], points[:, 1], color="red")
    ax.set_title("n=%d"%len(points))
    plt.show(block=block)

if __name__ == "__main__":
    n=20 #number of points
    points = generatePloygon(n)
    image = np.ones((400,400))*255
    hull = cv2.convexHull(points)
    # plotPolygon(points,True)
    rect = cv2.boundingRect(hull)
    
    print rect
    print hull
    cv2.drawContours(image,[rect],0,(0,0,255),5)
    K = n/2
    # plotPolygon(rect,True)
    plt.imshow(image,'gray')
    plt.show()
    # cv2.imshow("polygon2", out)