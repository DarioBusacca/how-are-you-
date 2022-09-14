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

Drive contenente i modelli pre-allenati, il file fer2013.csv ed il file data.h5: [drive](https://drive.google.com/drive/folders/1Po7uqMJ4h6-bmLjkRgXph1rGs7-xzrLV).
Il file _FER_training.ipynb_ è direttamente collegato a questa cartella drive.

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

Il training è stato eseguito su google colab.

__Nel modello presente sul drive l'accuratezza è al 68.153% dopo 50 epoche._

Eseguendo il file fer_confusion_matrix.py è anche possibile visualizzare la matrice di confusione normalizzata, il file verrà salvato all'interno della cartella **FER2013_VGG19**.

## Riconoscimento tramite foto

Per avviare il riconoscimento tramite foto, inserire un'immagine nella cartella **images**, e scrivere nel file _run.py_ il nome dell'immagine, a riga 27 in corrispondenza della dicitura _NOMEFILE_.
> python run.py

Il programma restituirà in output un file png dentro la cartella **images/results** con il risultato desiderato.

## Riconoscimento tramite video
Per avviare il riconoscimento tramite video, inserire il video nella cartella **video**, e modificare il file _face_detector.py_ nelle righe 81, 82 e 83, inserendo il nome del file al posto della dicitura NOMEFILE (il video deve necessariamente essere in mp4).
> python face_detector.py

Il programma farà partire il video, mostrando un rettangolo attorno alla faccia del soggetto inquadrato e l'evoluzione delle emozioni da quella iniziale a quella finale.

---

__L'idea di questa seconda parte del progetto è quella di riconoscere l'evoluzione delle emozioni da una live foto fatta con iPhone. Il video in questione infatti deve essere breve, più o meno il tempo che si impiega per scattarsi una foto tramite la fotocamera interna. Un consiglio è quello di salvare una live foto come video tramite il proprio iPhone (si può fare direttamente dalla galleria del cellulare), e di inserirla nella cartella del programma in formato mp4.__


