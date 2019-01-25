scores = model.evaluate([data_load.X_test, data_load.y_test2], data_load.y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
translation=np.chararray((2161,70),unicode=True) #number of prediction , their lenght
predictions = model.predict([X_test, y_test2], batch_size=16)
print(predictions.shape)
#print(predictions)

pred = K.argmax(predictions,axis=2) #argmax outputs index of maximum . oppsite of to_categorical
pred_val = K.eval(pred) #Tensor to array.
print(pred_val.shape)
print(translation.shape)

i=0
for p in pred_val:
    j=0
    let=0
    for number in p:

        if 27 <= number <= 52: #lower letters 27 to 52
            w = int(number+70)
            let= chr(w)

        elif 1 <= number <= 26: #upper letters 1 to 26
            w = int(number+64)
            let=chr(w)

        elif 53 <= number <= 62: #numbers 53 to 62
            w = int(number-5)
            let= chr(w)

        elif 63 <= number <= 79:
            let = data_load.symbols_inverse[number]
        else:
            let=""

        number=0
        translation[i][j]=let
        j=j+1
    i=i+1


for line in translation:
  print(line)