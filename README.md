# __HOW ARE YOU__

Questo programma è una implementazione pytorch basata su una CNN che riconosce le emozioni da una foto o da un video. 
Il file _run.py_ implementa il riconoscimento tramite una foto, mentre _face_detector.py_ quello tramite video.

## Dependencies

 - Python 2.7 o successive
 - Pytorch 0.2.0 o successive
 - h5py
 - sklearn
 - matplotlib
 - opencv

## Link utili

Drive contenente i modelli pre-allenati, il file fer2013.csv ed il file data.h5: [drive](https://drive.google.com/drive/folders/1Po7uqMJ4h6-bmLjkRgXph1rGs7-xzrLV)

Link al dataset ufficiale di FER2013, fornito da kaggle: [dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

## FER2013
Per eseguire il programma è necessaria la disponibilità di cuda. Il file _fer.py_ analizza data.h5. Il file data.h5 viene a sua volta creato da _preprocess_FER.py_, che esegue il preprocessamento del file csv scaricato da Kaggle.

- Il file _fer2013.csv_ va messo dentro la cartella **data**
- Il file _data.h5_ va messo dentro la cartella **data**
- I file dei modelli pre-allenati vanno messi dentro la cartella **FER2013_VGG19**

#### TRAINING
Il file da eseguire per fare il preprocessamento e iniziare il training dei modelli è _main.py_. Le opzioni per eseguirlo sono le seguenti:
- Iniziare il training dall'epoca 0, nel caso in cui esistano già dei modelli verranno formattati:
> python main.py
- Iniziare il training dall'ultimo salvataggio, che corrisponderà all'epoca con la miglior percentuale di accuratezza nel PrivateTest_model
> python main.py -r
- Cambiare il learning rate (default 0.01) o la batch size (default 8)
> python main.py -lr _VALUE_
> python main.py -bs _VALUE_

__Nel modello presente sul drive l'accuratezza è al 58.707% dopo 16 epoche, ma può raggiungere valori molto superiori se si hanno a disposizione i mezzi fisici.__

Eseguendo il file fer_confusion_matrix.py è anche possibile visualizzare la matrice di confusione normalizzata, il file verrà salvato all'interno della cartella **FER2013_VGG19**.

## Riconoscimento tramite foto

Per avviare il riconoscimento tramite foto, inserire un'immagine nella cartella **images**, e scrivere nel file _run.py_ il nome dell'immagine, a riga 27 in corrispondenza della dicitura _NOMEFILE_.
> python run.py

Il programma restituirà in output un file png dentro la cartella **images/results** con il risultato desiderato.




