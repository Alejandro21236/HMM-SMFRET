This project is predicting smFRET data in dynamicallt hetereogeneous systems like CAS9, using conformational MD data.
Within Each Folder contains two CSVs containing original trajectory data analyzed from each conformational state as provided by Palermo et. al.
cas9data.py is the script to extract the data from the files. there are copies of this script in each file.
CAS9_FRET.py is the LSTM. hmm.py is the Hidden Markov Models. 
APO.csv, DNA.csv, etc. are the extracted data files, Cas9_fret.csv is appended with the statistical features.
The Second CSV is appended with statistical features of the simulated smFRET data.
OLSregression.xlsx contains the AIC and BIC information.
The python scripts and their versions contain the LSTM, HMM, and statistical analyses used in the paper.
The  data extraction scripts are also available in each folder.
