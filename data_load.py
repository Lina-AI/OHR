import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Flatten
from keras.utils import to_categorical
import html
from numpy import newaxis
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence


X=[]
Y=[]

line_count=0
n=0
d=0
for root, directories, filenames in os.walk('C:\\Users\\Sultan\\Lina\\lina\\lina\\hwr-master\\original'): #dataset path on machine
    for filename in filenames:
        
##START LABEL READ FROM DATASET
        d=d+1
        a=os.path.join(root,filename)
        tree=ET.parse(a)
        r = tree.getroot()


        nums=[]
        k=0
        o=0

        alpha='abcdefghijklmnopqrstuvwxyz'
        numbers='0123456789'
        symbols = {':': 63, '"': 64, '%': 65, '/': 66, '.': 67, '!': 68,
                   '?': 69, '(': 70, ')': 71, '#': 72, ';': 73, ',': 74, "'": 75, '&': 76, '+': 77, '-': 78, ' ':79}
        symbols_inverse = {63:':', 64: '"',65 : '%', 66: '/',  67:'.', 68:'!',
                   69:'?',  70:'(', 71:')', 72:'#', 73:';', 74:',', 75:"'", 76:'&',  77:'+', 78:'-', 79:' '}
        SOS_token = 0
        EOS_token = 80
        j=0
        trans = r.find("Transcription")
        for i, textline in enumerate(r.iter('TextLine')):
            o=o+1

        words=np.zeros((o,70))


        for i, textline in enumerate(r.iter('TextLine')):
            text = textline.get('text')
            f=html.unescape(text) #converts the &amp;quote to "
            k=k+1
            l=0
            for word in f:
               for z in word:
#                    _+=1
                    if z in alpha.lower(): #lower letters 27 to 52
                        let=(abs(ord(z)-70))

                    elif z in alpha.upper(): #upper letters 1 to 26
                        let=(abs(ord(z)-64))

                    elif z in numbers: #numbers 53 to 62
                        let = ord(z)+5

                    elif z in symbols.keys():
                        let = symbols[z]
                    else:
                        raise Exception('Unexpected char while parsing dataset: {}'.format(z))
                    words[j][l] = let
                    l+=1
            j+=1
##END LABEL READ FROM DATASET




##START INPUT READ FROM DATASET
        stroke_set = r.find("StrokeSet")

        strokes = []
        strokes_div=[] #divided strokes
        maxmin=[]
        maxmin2=[]
        maxminy=[]
        time_points=[]



        for stroke_node in stroke_set:
            
            maxmin.append([]) #for x points
            maxminy.append([])  #for y points
            maxmin2.append([])
            time_points.append([])  #for time
            strokes_div.append([])
            
            for point in stroke_node:

                x = int(point.attrib['x'])
                x2 = int(point.attrib['x'])
                y = int(point.attrib['y'])
                t = float(point.attrib['time'])
                #strokes[-1].append((x,y))
                maxmin[-1].append(x)
                maxminy[-1].append(y)
                maxmin2[-1].append(x2)
                time_points[-1].append(t)
                strokes_div[-1].append((x,y))
                strokes.append((x,y))
        
##END INPUT READ FROM DATASET
        
#START NORMALIZE
        x_elts = [x[0] for x in strokes]
        y_elts = [x[1] for x in strokes]
        #print(x_elts)
        current_Xmax = max(x_elts)
        current_Xmin = min(x_elts)
        current_Ymax = max(y_elts)
        current_Ymin = min(y_elts)
        #print(current_Xmax)
        #print(current_Ymax)
        x_list=[]
        y_list=[]
        for x_point in x_elts:
            x_point=((x_point - current_Xmin)/(current_Xmax-current_Xmin))
            x_list.append(x_point)
        #print(x_list)
        for y_point in y_elts:
            y_point=((y_point - current_Ymin)/(current_Ymax-current_Ymin))
            y_list.append(y_point)

        #print(y_list)
        combined=[]
        combined= list(zip(x_list , y_list))
#END NORMALIZE
        
        g=1
        c=np.asarray(combined)
        c=np.insert(c, 2, 0, axis=1) #adds third('2') column('axis=1') of zeros
        count=0
        leni=0
        ind3=0
        ind4=0
 
#NORMALIZE MAXMIN2 &MAXMIN3 TO COMPARE THEM TO KNOW NEW LINE
        for x in maxmin2:
            for u in x:
                maxmin2[ind3][ind4]=((u - current_Xmin)/(current_Xmax-current_Xmin))
                ind4+=1

            ind4=0
            ind3+=1
            

        
##END
        maxmin3 = maxmin2[1:]
      
        for i, j in zip(maxmin2, maxmin3):
            count+=1
            leni+=len(i)

            if abs(max(i)-min(j))>0.4:
                    c[leni,2]=1
                    g+=1
        
                
#START MARK STOKE
        c=np.insert(c, 3, 0, axis=1) #adds fourth('3') column('axis=1') of zeros
        points_count=0
        a3=[] #created just to fix some bug
        a3.append((0,0))

        for stroke_val ,tri_val in zip(strokes_div,c):

            for stroke_ind2, stroke_val2 in enumerate(stroke_val):
                if stroke_ind2==0:
                    if stroke_val2 != a3[-1]:
                        a3.append(stroke_val2)
                        c[points_count,3]=1
                
                
                elif stroke_val2==stroke_val[-1]:
                    if stroke_val2 != a3[-1]:
                        a3.append(stroke_val2)
                        c[points_count,3]=2 
                points_count+=1
#END MARK STROKE
                

        p=0
        f=0
        h=0
        tri=np.zeros((g,2000,3))
        for i in c:
            if i[2]==0:
                tri[h,f,0]=i[0]
                tri[h,f,1]=i[1]
                tri[h,f,2]=i[3]
                f=f+1
            else:
                tri[h,f,0]=i[0]
                tri[h,f,1]=i[1]
                tri[h,f,2]=i[3]
                f=0
                h=h+1
        if g==k:
            for i in tri:
                X.append(i)
            for j in words:
                Y.append(j)

Y=np.asarray(Y)
Y_cat = to_categorical(Y, num_classes=None)
X=np.asarray(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.2, random_state=1)

y_train2=np.zeros(y_train.shape)
for i, target_text in enumerate(y_train):
    for t, char in enumerate(target_text):
        if t < 69:
	        y_train2[i,t+1] = char

y_test2=np.zeros(y_test.shape)
for i, target_text in enumerate(y_test):
    for t, char in enumerate(target_text):
        if t < 69:
	        y_test2[i,t+1] = char


print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
print('done')
