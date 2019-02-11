# Online-Handwriting-Recognition-using-Encoder-Decoder-model

Keras implementation of a sequence to sequence model for online handwriting recognition using an encoder-decoder architecture

### Requirements

* TensorFlow 1.10.0
* Keras 2.2.2
* [Anaconda](https://www.anaconda.com/distribution/)

### Dataset
The dataset used is the [IAM On-Line Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database). It contains different sentences acquired
by writers varying in ages, genders and native languages with all of this info and more
stored in the database. The IAM-OnDB contains forms of unconstrained handwritten
english text acquired on a whiteboard with the E-Beam System, all texts in the database
are built using sentences provided by the LOB Corpus. The collected data is stored in
xml-format, including the writer-id, the transcription and the setting of the recording

## Data Preparation

### Input Data

![alt text](https://github.com/AbeerEisa/Online-Handwriting-Recognition-using-Encoder-Decoder-model/blob/master/images/inputdata.jpg)

#### Preprocessing
A normalizing function was applied to the X and Y points to adjust the values measured on different scales to a notionally common scale. This process is done for each file individually so that the sequences will fit into a [0,1] [0,1] bounding box without affecting the original aspect ratio.

```
        x_elts = [x[0] for x in strokes]
        y_elts = [x[1] for x in strokes]
        current_Xmax = max(x_elts)
        current_Xmin = min(x_elts)
        current_Ymax = max(y_elts)
        current_Ymin = min(y_elts)
        scale = 1.0 / (current_Ymax - current_Ymin)  
        
        #print(current_Xmax)
        #print(current_Ymax)
        x_list=[]
        y_list=[]
        for xx_point in x_elts:
            xx_point=((xx_point - current_Xmin)/(current_Xmax-current_Xmin))
            x_list.append(xx_point)
        for yy_point in y_elts:
            yy_point=((yy_point - current_Ymin)/(current_Ymax-current_Ymin))
            y_list.append(yy_point)
            
        combined=[]
        combined= list(zip(x_list , y_list))
```

![alt text](https://github.com/AbeerEisa/Online-Handwriting-Recognition-using-Encoder-Decoder-model/blob/master/images/pre.PNG)

#### Segmentation 
The input text was divided into lines with maximum lenght of 2000.

![alt text](https://github.com/AbeerEisa/Online-Handwriting-Recognition-using-Encoder-Decoder-model/blob/master/images/pre2.PNG)

#### Feature Extraction

```
        sin_list = []
        cos_list = []
        x_sp_list = []
        y_sp_list = []
        pen_up_list = []
        writing_sin = []
        writing_cos = []
        
        
        for stroke in r.findall('StrokeSet/Stroke'):
            x_point, y_point, time_list = [], [], []
            
            for point in stroke.findall('Point'):
                x_point.append(int(point.get('x')))
                y_point.append(int(point.get('y')))
                if len(time_list) == 0:
                    first_time = float(point.get('time'))
                    time_list.append(0.0)
                else:
                    time_list.append(
                                float(point.get('time')) - first_time)
            
            # calculate cos and sin
            x_point[:] = [(point - current_Xmin) * scale for point in x_point]
            y_point[:] = [(point - current_Ymin) * scale for point in x_point]
            
            

            angle_stroke = []
            if len(x_point) < 3:
#                print("Oh no",len(x_point))
                for _ in range(len(x_point)):
                    sin_list += [0]
                    cos_list += [1]
            else:
                for idx in range(1, len(x_point) - 1):
                    x_prev = x_point[idx - 1]
                    y_prev = y_point[idx - 1]
                    x_next = x_point[idx + 1]
                    y_next = y_point[idx + 1]
                    x_now = x_point[idx]
                    y_now = y_point[idx]
                    p0 = [x_prev, y_prev]
                    p1 = [x_now, y_now]
                    p2 = [x_next, y_next]
                    v0 = np.array(p0) - np.array(p1)
                    v1 = np.array(p2) - np.array(p1)
                    angle = np.math.atan2(
                                np.linalg.det([v0, v1]), np.dot(v0, v1))
                    angle_stroke.append(angle)
                new_angle_stroke = [0] + angle_stroke + [0]
                sin_stroke = np.sin(new_angle_stroke).tolist()
                cos_stroke = np.cos(new_angle_stroke).tolist()
                sin_list += sin_stroke
                cos_list += cos_stroke
                    
            # calculate speed
            if len(x_point) < 2:
                    for _ in range(len(x_point)):
                        x_sp_list += [0]
                        y_sp_list += [0]

                    if len(x_point) < 1:
                        print("Meet 0")
                        exit()
                    x_sp = [0]
                    y_sp = [0]

            else:
                    time_list = np.asarray(time_list, dtype=np.float32)
                    time_list_moved = np.array(time_list)[1:]
                    time_diff = np.subtract(
                        time_list_moved, time_list[:-1])
                    for idx, v in enumerate(time_diff):
                        if v == 0:
                            time_diff[idx] = 0.001
                    x_point_moved = np.array(x_point)[1:]
                    y_point_moved = np.array(y_point)[1:]
                    x_diff = np.subtract(x_point_moved, x_point[:-1])
                    y_diff = np.subtract(y_point_moved, y_point[:-1])
                    x_sp = np.divide(x_diff, time_diff).tolist()
                    y_sp = np.divide(y_diff, time_diff).tolist()
                    x_sp = [0] + x_sp
                    y_sp = [0] + y_sp
                    x_sp_list += x_sp
                    y_sp_list += y_sp
            # pen up and down
            pen_up = [1] * (len(x_point) - 1) + [0]
            pen_up_list += pen_up
            # writing direction
            w_sin_stroke = []
            w_cos_stroke = []
            for idx, x_v in enumerate(x_sp):
                y_v = y_sp[idx]
                slope = np.sqrt(x_v * x_v + y_v * y_v)
                if slope != 0:
                    w_sin_stroke.append(y_v / slope)
                    w_cos_stroke.append(x_v / slope)
                else:
                    w_sin_stroke.append(0)
                    w_cos_stroke.append(1)
            writing_sin += w_sin_stroke
            writing_cos += w_cos_stroke
```

### Label Data
The label data is initially represented as samples of lines that consist of ASCII characters. The lines have a fixed length of 70, based on the line that has the maximum number of characters which is found to be 70, and the lines with less characters are padded with ‘zeros’ at the end. The label data is converted to hexadecimal ASCII code; uppercase letters, lowercase letters, and numbers represented as hexadecimals values from (65 to 90), (97 to 122) and (48 to 57) respectively.

Next, the labels are converted to a simpler representation as integers from 1 to 80 in which the uppercase letters, lowercase letters and numbers are represented as integers from (1 to 26), (27 to 52) and (53 to 62) respectively. Other ASCII symbols are represented as values from 62 to 80. 0 is reserved for the padding values.

Finally, the data is converted to One-hot Encoding representation, which is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction. Each integer is represented as a vector of length ‘80’ with a single high (1) bit and all the others low (0) bit.

## Recognition

### Encoder-Decoder Architecture
Encoder-Decoder models are architectures for sequence-to-sequence (Seq2Seq) 
prediction problems. Sequence-to-sequence is a problem setting where the input is a
sequence and the output is also a sequence. Encoder-decoder models are mostly used for
language to language translation problems.

The system is comprised of two sub-models, as the name suggests: Encoder: to
encode the source sequence, and Decoder: to decode the encoded source sequence into
the target sequence

The model estimates the conditional probability p(y1, ... ,yi|x1, ... ,xj), where (x1, ... , xj)
is an input sequence and y1, ..., yi is its corresponding output sequence whose length i
may differ from j.

Encoder: An LSTM layer is used as the encoder, its goal is to step through the input
time steps and encode the entire sequence into fixed length vector representation v of
the input sequence (x1, ..., xj). This vector representation is called the context vector(c)
and is obtained from the last hidden layer of the encoder LSTM as shown in the figure.

Decoder: Another LSTM layer is used as the decoder and is responsible for stepping
through the output time steps and computing the probability of (y1, ..., yi) while reading
from the context vector. The context vector is set as the initial hidden state of the LSTM
layer.

Each p(yi|jc, y1, ... yi−1) distribution in the equation is represented with a softmax
layer over all the letters in the alphabet.

![alt text](https://github.com/AbeerEisa/Online-Handwriting-Recognition-using-Encoder-Decoder-model/blob/master/images/encdec.PNG)

### Attention
Before Attention mechanism, the recognition relied on reading input sequence of a line
and compressing all information into a fixed-length vector ( the context vector). However,
if the input sequence is too long, it leads to information loss therefore low recognition
accuracies.. This can be imagined as a sentence with hundreds of words represented by
only several words.

However, attention partially fixes this problem by allowing the model to look over
all the information the input holds, and then focus on specific parts of the input sequence.

Instead of the hidden state being obtain from only the last hidden layer in the
encoder, a hidden state for each input time-step is gathered from the encoder, it is then
scored and normalized using a softmax function to be a probability over the encoder
hidden states, next the probabilities are used to calculate a weighted sum of the encoder
hidden states to provide a context vector to be used in the decoder. Finally, the context
vector and the target decoder are concatenated together and passed through a softmax
layer to predict the probability of the next letter in the sequence.

![alt text](https://github.com/AbeerEisa/Online-Handwriting-Recognition-using-Encoder-Decoder-model/blob/master/images/att.PNG)

## Results
Accuracy of the system is 96.036.

#### example 1
Label: [’A’ ’c’ ’k’ ’n’ ’o’ ’w’ ’l’ ’e’ ’d’ ’g’ ’m’ ’e’ ’n’ ’t’ ’s’ ’.’ ” ’A’ ” ’g’ ’r’ ’a’ ’p’ ’h’ ’i’ ’c’ ’a’ ’l’]

![alt text](https://github.com/AbeerEisa/Online-Handwriting-Recognition-using-Encoder-Decoder-model/blob/master/images/g2.PNG)

Recognized output: [’A’ ’c’ ’k’ ’n’ ’o’ ’w’ ’l’ ’e’ ’d’ ’n’ ’m’ ’e’ ’n’ ’t’ ’s’ ’.’ ” ’I’ ” ’g’ ’r’ ’a’
’p’ ’h’ ’i’ ’c’ ’a’ ’l’]


#### example 2
Label: [’B’ ’o’ ’a’ ’r’ ’d’ ’?’ ’"’ ” ’"’ ’Y’ ’e’ ’s’ ’,’ ” ’s’ ’i’ ’r’ ’.’ ’"’]

![alt text](https://github.com/AbeerEisa/Online-Handwriting-Recognition-using-Encoder-Decoder-model/blob/master/images/g3.PNG)

Recognized output: [’B’ ’o’ ’a’ ’r’ ’d’ ’,’ ’"’ ” ’"’ ’W’ ’e’ ’s’ ’,’ ” ’s’ ’i’ ’r’ ’.’ ’"’]


#### example 3
Label: [’p’ ’o’ ’i’ ’n’ ’t’ ” ’a’ ’t’ ” ’w’ ’h’ ’i’ ’c’ ’h’ ” ’h’ ’e’ ” ’s’ ’w’ ’e’ ’r’ ’v’ ’e’ ’s’]

![alt text](https://github.com/AbeerEisa/Online-Handwriting-Recognition-using-Encoder-Decoder-model/blob/master/images/g4.PNG)

Recognized output: [’p’ ’o’ ’i’ ’n’ ’t’ ” ’a’ ’t’ ” ’w’ ’h’ ’i’ ’c’ ’h’ ” ’h’ ’e’ ” ’s’ ’w’ ’e’ ’r’ ’v’
’e’ ’s’]

## References
["A ten-minute introduction to sequence-to-sequence learning in Keras"](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html) by François Chollet.

## Authors
* [Abeer Eisa](https://github.com/AbeerEisa)
* [Lina Abdelkarim]()
