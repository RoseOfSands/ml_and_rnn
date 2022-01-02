import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class Anomaly:
  def __init__(self, path):
    self.file_format = path.split('.')[-1]
    self.path=path
  def read_file(self): 
    if self.file_format=='xls' or self.file_format=='xlsx':
        data = pd.read_excel(self.path, sep=',', encoding='cp1251')
    else:
        data = pd.read_csv(self.path, sep=',', encoding='cp1251')
    return(data)

  def find_anomalies(self, data, column_name):
      X=data[[column_name]]
      std_dev=X.std(axis=0)
      anomalies=[]
      lines=[]
      n=0
      X=np.array(X)
      for i in X:
          n=n+1
          if float(i)>float(3*std_dev):
              lines.append(n)
              anomalies.append(float(i))
      return (lines, anomalies)
  
  def visualization(self, data, lines, anomalies):
      plt.scatter([j for j in range (len(data))],[data])  
      plt.scatter(lines,anomalies,c=["r"])
  def delete_anomalies(self, data, lines):
      data.drop(labels=[i for i in lines], axis=0, inplace=True)
      return (data)
  def write(self, data, out_path):  
      data=pd.DataFrame(data)
      data.to_csv(out_path, sep='\t', encoding='cp1251', index=False)
