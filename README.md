# Manuale per l'uso del programma

## Struttura
Il programma è strutturato in più file:

perceptron.py : 

                file contenente la classe Perceptron ed i metodi training(X, Y){} e predict(xi){}, 
                con X array di elementi xi (punto in R^len(xi) dimensioni) e Y array che ne identifica la classe.
                
voted_perceptron.py : 

                file contenente la classe Voted-Perceptron ed i metodi training(X, Y){} e predict(xi){}, 
                con X array di elementi xi (punto in R^len(xi) dimensioni) e Y array che ne identifica la classe.

index.py :  
  
                file dove avviene il procedimento di raccolta dati dai Datasets scaricati, la loro suddivisione 
                in insiemi Train e Test e la successiva sottomissione dei dati Train e successivamente Test ai 
                due algoritmi precedentemente citati, come previsto dal metodo di Cross Validation.

csv file : 

                Datasets che ho scelto dal sito di http://mldata.org/ per testare le prestazini degli algoritmi.

## Funzionamento
Nella classe principale dove si svolgono tutti i test ( index.py ) vengono inizialmente specificati i file .csv da dove 
si recuperano i dati. I dati sono quindi suddivisi in "values" e "labels", nonchè punti in R^n dimensioni e classe alla 
quale il punto appartiene (supporremo solo problemi di classificazioni binaria, dunque classe 1 o -1). 
Viene a questo punto usata la funzione train_test_split dal modulo sklearn.model_selection che divide il dataset in 
Train e Test, con un'opportuna percentuale data (per gli esempi si utilizza 90% Train e 10% Test).

Viene dunque creato l'oggetto Perceptron e "allenato" con il Train, in modo tale che trovi il primo iperpiano che classifica
tutti gli esempi. Il primo output è quindi l'Iperpiano che viene trovato e si indica al lato il numero di cicli fatti su 
tutti gli elementi del Train per capire se l'iperpiano generato è quello che soddisfa tutti i punti (e quindi il Datasets 
è linearmente separabile) o è solamente l'ultimo iperpiano calcolato prima del raggiungimento del limite delle epoche 
(in questo caso non è possibile specificare se il dataset è o meno linearmente separabile).

Lo stesso procedimento è svolto con l'algoritmo Voted Perceptron, e vengono forniti anche a questo algoritmo gli stessi dati 
di Train e Test.

Infine sono richiamati i metodi confusion_matrix() , accuracy_score() e classification_report() per dare una panoramica 
sull'accuratezza con la quale sono stati classificati gli elementi del Test.

## Aggiungere un Datasets
E' possibile modificare il programma per aggiungere a piacimento un dataset. Occorre un file .csv contenente i dati, 
e appositamente situato nella cartella '/csv'. Successivamente si procede a estrarre dal file .csv i rispettivi dati e 
costruire con questi un array pandas X. Dobbiamo fare ancora un'ultimo passaggio, quello di rimuovere dall'array pandas (X) la
riga che corrisponde all'informazione sulla classe di appartenenza dei dati, ed inserirla in un array apposito (Y).
A questo punto il programma è capace di trattare qualsiasi insieme di dati (purchè numerico) per mostrare la matrice di 
confusione e l'accuratezza.

N.B. E' necessario conoscere la struttura del dataset, i nomi dei campi ed il significato dei dati, per un corretto 
utilizzo del programma.

## Comprendere l'output
L'output è diviso in due componenti: il Test sull'algoritmo Perceptron, ed il test sul Voted Perceptron.
Per entrambi i casi vengono mostrate le stesse informazioni:
- Singol Layer accuracy:

        indica l'accuratezza nella classificazione dei Test rispetto alla vera classe a cui appartengono i dati;
        
- Confusion matrix, without normalization: 

        indica la matrice di confusione con il numero di test positivi e negativi corretti e falsi;
        
- Normalized confusion matrix:

        indica la matrice di confusione precedente, ma mostrata in base alle percentuali;
        
- Classification report:

        indica i valori di "precision, recall, f1-score, support" per entrambe le classi di tipologie.
        
# Riferimenti
Ringrazio [UCI](http://archive.ics.uci.edu/ml/index.php) perchè descrive moltissimi datasets che sono forniti da [mldata](http://mldata.org/), rendendo comprensibile il significato dei dati. 
Ringrazio per avermi reso piu' chiaro il Perceptron ed il Voted Perceptron con esempi, implementazioni, spiegazioni e video i seguenti autori e pagine:
- [Packt>](https://www.youtube.com/channel/UC3VydBGBl132baPCLeDspMQ)
- [Bgmee on github](https://github.com/bmgee)
- [Jason Brownlee](https://machinelearningmastery.com/about/)
- [StackOverflow](https://stackoverflow.com/)
- [Freund & Schapire 1999](https://link.springer.com/content/pdf/10.1023/A:1007662407062.pdf)
- [Wikipedia](https://en.wikipedia.org/wiki/Perceptron)
