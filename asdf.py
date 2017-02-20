from skimage.io import imread
from scipy import ndimage
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from vector import distance, pnt2line
from sklearn.datasets import fetch_mldata
from keras.layers.core import Activation, Dense
from keras.models import Sequential,load_model
from keras.optimizers import SGD
from skimage.color import rgb2gray
from skimage.measure import label, regionprops


def to_categorical(labels, n):
    retVal = np.zeros((len(labels), n), dtype='int')
    ll = np.array(list(enumerate(labels)))
    retVal[ll[:,0],ll[:,1]] = 1
    return retVal


def padwithtens(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

def findLine(frame):
    print "FUNKCIJA"
    lower = np.array([230,0,0])
    upper = np.array([255,255,255])

    newFrame = cv2.inRange(frame, lower, upper)

    maxLineGap = 20
    minLineLength = 5

    lines = cv2.HoughLinesP(newFrame, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    print lines
    line = lines[0]

    first = []
    second = []
    third = []
    fourth = []
    for arrayM in lines:
        for array in arrayM:
            first.append(array[0])
            second.append(array[1])
            third.append(array[2])
            fourth.append(array[3])


    min = 1500;
    ind = 0
    for hh in range(0,len(first)):
        if first[hh] < min:
             min = first[hh]
             ind = hh

    max = 0
    inx = 0
    for hh in range(0,len(first)):
        if third[hh] > max:
             max = third[hh]
             inx = hh


    x1 = first[ind]
    y1 =second[ind]
    x2 = third[inx]
    y2 = fourth[inx]
    # x1 = line[0,0]
    # y1 = line[0,1]
    # x2 = line[0,2]
    # y2 = line[0,3]

    return x1,y1,x2,y2


cc = -1
def nextId():
    global cc
    cc += 1
    return cc


def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = distance(item['center'], obj['center'])
        if(mdist<r):
            retVal.append(obj)
    return retVal






kernel = np.ones((2,2),np.uint8)
lower = np.array([230, 230, 230])
upper = np.array([255, 255, 255])


fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('C:\\Users\\Stefan\\Desktop\\Videos\\videos\\rv.avi',fourcc, 20.0, (640,480))
putanja = "C:\\Users\\Stefan\\Desktop\\Videos\\slike"


elements = []
t =0
counter = 0
times = []
z = 0;

prelazeci = []
momenti = []

if __name__ == "__main__":
    f_name = 'Videos/video-0.avi'
    video = cv2.VideoCapture(f_name)


    videoOn = True
    t = 0
    while videoOn:
        start_time = time.time()
        videoOn, img = video.read()

        if t==0:
            x1, y1, x2, y2 = findLine(img)
            print x1,y1,x2,y2
            line = [(x1,y1), (x2,y2)]

#        obradafunkcije(line,frame)

        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(img, lower, upper)
        img0 = 1.0 * mask

        img0 = cv2.dilate(img0, kernel)  # cv2.erode(img0,kernel)
        img0 = cv2.dilate(img0, kernel)

        file_name ='slike/video-'  + str(t) + '.png'
        cv2.imwrite(file_name, img0)

        labeled, nr_objects = ndimage.label(img0)
        objects = ndimage.find_objects(labeled)
        for i in range(nr_objects):
            loc = objects[i]
            (xc, yc) = ((loc[1].stop + loc[1].start) / 2,
                        (loc[0].stop + loc[0].start) / 2)
            (dxc, dyc) = ((loc[1].stop - loc[1].start),
                          (loc[0].stop - loc[0].start))

            if (dxc > 11 or dyc > 11):
                cv2.circle(img, (xc, yc), 16, (25, 25, 255), 1)
                elem = {'center': (xc, yc), 'size': (dxc, dyc), 't': t}
                # find in range
                lst = inRange(20, elem, elements)
                nn = len(lst)
                if nn == 0:
                    elem['id'] = nextId()
                    elem['t'] = t
                    elem['pass'] = False
                    elem['history'] = [{'center': (xc, yc), 'size': (dxc, dyc), 't': t}]
                    elem['future'] = []

                    elements.append(elem)
                elif nn == 1:
                    lst[0]['center'] = elem['center']
                    lst[0]['t'] = t
                    lst[0]['history'].append({'center': (xc, yc), 'size': (dxc, dyc), 't': t})
                    lst[0]['future'] = []

        for el in elements:
            tt = t - el['t']
            if (tt < 3):
                dist, pnt, r = pnt2line(el['center'], line[0], line[1])
                c = (25, 25, 255)
                if r > 0:

                    cv2.line(img, pnt, el['center'], (0, 255, 25), 1)
                    c = (25, 25, 255)
                    if (dist < 9):  # 9
                        c = (0, 255, 160)
                        if el['pass'] == False:
                            el['pass'] = True
                            print el['center']
                            momenti.append(t)
                            counter += 1
                            prelazeci.append(el)
                            print t

                cv2.circle(img, el['center'], 16, c, 2)

                # if t == elem['moment'] + 20:
                #     prelazeci.append(img0)

                id = el['id']
                cv2.putText(img, str(el['id']),
                            (el['center'][0] + 10, el['center'][1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                for hist in el['history']:
                    ttt = t - hist['t']
                    if (ttt < 100):
                        cv2.circle(img, hist['center'], 1, (0, 255, 255), 1)

                for fu in el['future']:
                    ttt = fu[0] - t
                    if (ttt < 100):
                        cv2.circle(img, (fu[1], fu[2]), 1, (255, 255, 0), 1)

        elapsed_time = time.time() - start_time
        times.append(elapsed_time * 1000)
        cv2.putText(img, 'Counter: ' + str(counter), (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)
        z += 1
        # print nr_objects
        t += 1

        cv2.imshow('frame', img)
        file_name = 'slike/video-'  + str(t) + '.png'
        cv2.imwrite(file_name, img)
        k = cv2.waitKey(30) & 0xff

        if t == 1199:
            break
        out.write(img)

    out.release()
    video.release()
    cv2.destroyAllWindows()

    et = np.array(times)
    print 'mean %.2f ms' % (np.mean(et))

    zz=0
    izvadjeni = []
    for el in prelazeci:
        for hi in el['history']:
            if hi['t']+4<t:
                if hi['t']+4 == momenti[zz]:
                    izvadjen = {'center': hi['center'], 't': hi['t']}
        zz+=1
        izvadjeni.append(izvadjen)




    putanja = 'slike/video-'
    kk = 0
    for el in izvadjeni:
        novaputanja = putanja + str(el['t']) + '.png'
        print novaputanja
        xi,yi = el['center']
        xi1 = xi-14
        yi1 = yi-14
        xi2 = xi+14
        yi2 = yi+14
        print str(xi),str(yi)
        slika = imread(novaputanja)
        novaslika = slika[yi1 : yi2, xi1 : xi2]
        put = 'noveslike/broj-'+str(kk) + '.png'
        cv2.imwrite(put,novaslika)
        # plt.imshow(slika)
        # plt.show()
        kk += 1


    model = load_model('model.h5')
    # digits = fetch_mldata('MNIST original', data_home='mnist')
    #
    # data = digits.data / 255.0
    # labels = digits.target.astype('int')
    #
    # train_rank = 10000
    # test_rank = 100
    #
    # train_subset = np.random.choice(data.shape[0], train_rank)
    # test_subset = np.random.choice(data.shape[0], test_rank)
    #
    # train_data = data
    # train_labels = labels
    #
    # trening = []
    # for index in range(0, len(train_data)):
    #     print 'Slika po redu: ' + str(index)
    #     image = train_data[index].reshape(28, 28)
    #     newImg = np.lib.pad(image, (30, 30), padwithtens)
    #
    #     imageJustNum = newImg[:, :] > 0
    #
    #     labeled_img = label(imageJustNum)
    #     regions = regionprops(labeled_img)
    #
    #     visina_sr = round((regions[0].bbox[0] + regions[0].bbox[2]) / 2)
    #     sirina_sr = round((regions[0].bbox[1] + regions[0].bbox[3]) / 2)
    #     DL1 = visina_sr - 14
    #     TR1 = visina_sr + 14
    #     DL2 = sirina_sr - 14
    #     TR2 = sirina_sr + 14
    #
    #     img = newImg[regions[0].bbox[0]: regions[0].bbox[2], regions[0].bbox[1]: regions[0].bbox[3]]
    #
    #     img_crop = newImg[int(DL1): int(TR1), int(DL2): int(TR2)]
    #     # plt.imshow(img_crop, 'gray')
    #     # plt.show()
    #
    #     trening.append(img_crop.reshape(784))
    #
    # trening = np.array(trening)
    #
    # print trening.shape
    # # test dataset
    # test_data = data[test_subset]
    # test_labels = labels[test_subset]
    #
    # train_out = to_categorical(train_labels, 10)
    # test_out = to_categorical(test_labels, 10)
    #
    # model = Sequential()
    # model.add(Dense(70, input_dim=784))
    # model.add(Activation('relu'))
    # model.add(Dense(30))
    # model.add(Activation('tanh'))
    # model.add(Dense(10))
    # model.add(Activation('softmax'))
    #
    # sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    # model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    #
    # training = model.fit(trening, train_out, nb_epoch=20, batch_size=400, verbose=1)
    # print training.history['loss'][-1]
    #
    # model.save('model.h5')

    suma = 0
    for ii in range(0,len(prelazeci)):
        putanja = 'noveslike/broj-'+str(ii) + '.png'
        imagePred = imread(putanja)
        prediction = model.predict(imagePred.reshape(1, 784), verbose=1)
        vrednost = np.argmax(prediction)
        print "Vrednost: " + str(vrednost)
        suma += vrednost


    print "Suma: "+ str(suma)