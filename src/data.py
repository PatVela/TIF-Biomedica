"""
Los datos provienen de 
https://physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm

Los registros se digitalizaron a 360 muestras por segundo y canal con una resolución de 11 bits en un intervalo de 10 mV.
Dos o más cardiólogos anotaron independientemente cada registro; los desacuerdos se resolvieron para obtener las anotaciones de referencia legibles por ordenador
para cada latido (aproximadamente 110.000 anotaciones en total) incluidas en la base de datos.

 Código Descripción
 N Latido normal (mostrado como . por el PhysioBank ATM, LightWAVE, pschart, y psfd)
 L Latido de bloqueo de rama izquierda del haz de His
 R Latido de bloqueo de rama derecha del haz de His
 B Latido de bloqueo de rama (no especificado)
 A Latido auricular prematuro
 a Latido auricular prematuro aberrado
 J Latido prematuro nodal (juncional)
 S Latido supraventricular prematuro o ectópico (auricular o nodal)
 V Contracción ventricular prematura
 r Contracción ventricular prematura R- on-T contracción ventricular prematura
 F Fusión de latido ventricular y normal
 e Latido auricular de escape
 j Latido nodal (juncional) de escape
 n Latido supraventricular de escape (auricular o nodal)
 E Latido ventricular de escape
 / Latido estimulado
 f Fusión de latido estimulado y normal
 Q Latido inclasificable
 ? Latido no clasificado durante el aprendizaje
"""

from __future__ import division, print_function
import os
from tqdm import tqdm
import numpy as np
import random
import h5py
from utils import *
from config import get_config

def preprocess( split ):
    nums = ['100','101','102','103','104','105','106','107','108','109','111','112','113','114','115','116','117','118','119','121','122','123','124','200','201','202','203','205','207','208','209','210','212','213','214','215','217','219','220','221','222','223','228','230','231','232','233','234']
    features = ['MLII', 'V1', 'V2', 'V4', 'V5'] 

    if split :
        testset = ['101', '105','114','118', '124', '201', '210' , '217']
        trainset = [x for x in nums if x not in testset]

    def dataSaver(dataSet, datasetname, labelsname):
        classes = ['N','V','/','A','F','~']#,'L','R',f','j','E','a']#,'J','Q','e','S']
        Nclass = len(classes)
        datadict, datalabel= dict(), dict()

        for feature in features:
            datadict[feature] = list()
            datalabel[feature] = list()

        def dataprocess():
          input_size = config.input_size 
          for num in tqdm(dataSet):
            from wfdb import rdrecord, rdann
            record = rdrecord('dataset/'+ num, smooth_frames= True)
            from sklearn import preprocessing
            signals0 = preprocessing.scale(np.nan_to_num(record.p_signal[:,0])).tolist()
            signals1 = preprocessing.scale(np.nan_to_num(record.p_signal[:,1])).tolist()
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(signals0, distance=150)

            feature0, feature1 = record.sig_name[0], record.sig_name[1]

            global lppened0, lappend1, dappend0, dappend1 
            lappend0 = datalabel[feature0].append
            lappend1 = datalabel[feature1].append
            dappend0 = datadict[feature0].append
            dappend1 = datadict[feature1].append
            # omitir un primer pico para tener suficiente alcance de la muestra 
            for peak in tqdm(peaks[1:-1]):
              start, end =  peak-input_size//2 , peak+input_size//2
              ann = rdann('dataset/'+ num, extension='atr', sampfrom = start, sampto = end, return_label_elements=['symbol'])
              
              def to_dict(chosenSym):
                y = [0]*Nclass
                y[classes.index(chosenSym)] = 1
                lappend0(y)
                lappend1(y)
                dappend0(signals0[start:end])
                dappend1(signals1[start:end])

              annSymbol = ann.symbol
              # eliminar parte de "N" que rompe el equilibrio del conjunto de datos 
              if len(annSymbol) == 1 and (annSymbol[0] in classes) and (annSymbol[0] != "N" or np.random.random()<0.15):
                to_dict(annSymbol[0])
        print("Procesando datos...")
        dataprocess()
        noises = add_noise(config)
        for feature in ["MLII", "V1"]: 
            d = np.array(datadict[feature])
            if len(d) > 15*10**3:
                n = np.array(noises["trainset"])
            else:
                n = np.array(noises["testset"]) 
            datadict[feature]=np.concatenate((d,n))
            size, _  = n.shape 
            l = np.array(datalabel[feature])
            noise_label = [0]*Nclass
            noise_label[-1] = 1
            
            noise_label = np.array([noise_label] * size) 
            datalabel[feature] = np.concatenate((l, noise_label))

        with h5py.File(datasetname, 'w') as f:
            for key, data in datadict.items():
                f.create_dataset(key, data=data)
        with h5py.File(labelsname, 'w') as f:
            for key, data in datalabel.items():
                f.create_dataset(key, data=data)        

    if split:
        dataSaver(trainset, 'dataset/train.keras', 'dataset/trainlabel.keras')
        dataSaver(testset, 'dataset/test.keras', 'dataset/testlabel.keras')
    else:
        dataSaver(nums, 'dataset/targetdata.keras', 'dataset/labeldata.keras')

def main(config):
    def Downloadmitdb():
        ext = ['dat', 'hea', 'atr']
        nums = ['100','101','102','103','104','105','106','107','108','109','111','112','113','114','115','116','117','118','119','121','122','123','124','200','201','202','203','205','207','208','209','210','212','213','214','215','217','219','220','221','222','223','228','230','231','232','233','234']
        for num in tqdm(nums):
            for e in ext:
                url = "https://physionet.org/physiobank/database/mitdb/"
                url = url + num +"."+e
                mkdir_recursive('dataset')
                cmd = "cd dataset && curl -O "+url
                os.system(cmd)

    if config.downloading:
        Downloadmitdb()
        #print("no descargar")
    return preprocess(config.split)

if __name__=="__main__":
    config = get_config()
    main(config)