import random
import numpy
import numpy as np
import pandas as pd
import tkinter as tk
import csv
from matplotlib import pyplot as plt
import seaborn as sns


class NeuronaSimple():
    def __init__(self, aprendizaje, x_total,error_permisible):
        self.aprendizaje = aprendizaje
        self.x_total = x_total
        self.error_permisible = error_permisible
        self.magnitud_error = error_permisible + 1
        self.lista_x, self.y_deseada =self.leer_csv() 
        self.lista_w = self.generate_w()
        self.primer_w = []
        self.w_incremental = []
        self.lista_pesos = []
        self.lista_errores=[]
        self.iteraciones = 0
        self.error_observado = []
        self.magnitudes_error = []
        self.aux = 1
        self.err=0
        self.err_nuevo=0
        self.neuronasimple()
        while self.magnitud_error > self.error_permisible:
            self.neuronasimple()
        graficaErr(self.magnitudes_error,self.iteraciones)
        versus(self.y_deseada, self.y_calculada)
        graficaPeso(self.lista_pesos,self.iteraciones)
        err_versus(self.y_deseada, self.iteraciones, self.error_observado)
        reporte(self.lista_pesos, self.error_permisible, self.magnitud_error, self.iteraciones, self.magnitudes_error)
    def neuronasimple(self):
        self.lista_u = self.u()
        self.y_calculada=self.activacion()
        self.error = self.error1()
        self.w_incremental = self.incremento_w()
        self.lista_w=self.w()
        self.iteraciones+= 1
        self.aux = self.aux + 1
        self.magnitud_error = numpy.linalg.norm(self.e)/(len(self.e))
        self.magnitudes_error.append(self.magnitud_error)
    def leer_csv(self):
        data = pd.read_csv('203454.csv') 
        sesgo=self.sesgo(data)
        sesgo=pd.DataFrame(data=sesgo)
        lista_x=pd.concat([sesgo,data[['X1','X2','X3']]],axis=1).values.tolist()
        lista_y=data[['Y']].values.tolist()
        arr_aux=[]
        for y in lista_y:
            arr_aux.append(y[0])
        return (numpy.array(lista_x), numpy.array(arr_aux))
    def generate_w(self):
        self.primer_w = lista_w = np.random.uniform(0, 1, self.x_total)
        return lista_w
    def incremento_w(self):
        self.w_incremental = self.aprendizaje * numpy.dot(self.error, self.lista_x)
        return self.w_incremental
    def w(self):
        nueva_w = self.lista_w + self.w_incremental
        self.lista_pesos.append(nueva_w)
        return nueva_w
    def u(self):
        lista_w_traspuesta = numpy.transpose(self.lista_w)
        lista_u = numpy.dot(self.lista_x, lista_w_traspuesta)
        return lista_u
    def activacion(self):
        return self.lista_u
    def error1(self):
        e = numpy.subtract(self.y_deseada,self.y_calculada)
        self.e = numpy.array(self.y_deseada) - numpy.array(self.y_calculada)
        self.error_observado.append(e)
        self.lista_errores.append(self.e)
        return self.e
    def sesgo(self, data):
        sesgo = numpy.ones((len(data),1))
        return sesgo


import tkinter as tk

class Interfaz:
    def __init__(self, master):
        self.master = master
        master.title("Interfaz con Tkinter")
        self.label_error = tk.Label(master, text="Margen de error")
        self.label_error.pack()
        self.entry_error = tk.Entry(master)
        self.entry_error.pack()
        self.label_aprendizaje = tk.Label(master, text="Aprendizaje")
        self.label_aprendizaje.pack()
        self.entry_aprendizaje = tk.Entry(master)
        self.entry_aprendizaje.pack()
        self.button_analizar = tk.Button(master, text="Analizar", command=self.analizar)
        self.button_analizar.pack()
        
    def interfaz(self):
        self.master.mainloop()

    def analizar(self):
        self.master.quit()
        NeuronaSimple(aprendizaje=float(self.entry_aprendizaje.get()), x_total=4, error_permisible=float(self.entry_error.get()))
        print("Margen de error:", self.entry_error.get())
        print("Aprendizaje:", self.entry_aprendizaje.get())


def graficaErr(error, valor_x):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots()
    ax.set_title('Gráfica evolución del error')
    ax.set_xlabel('Iteración')
    ax.set_ylabel('Magnitud del error')
    ax.plot(numpy.arange(valor_x), error, label='Error')
    ax.legend(loc='upper right')
    plt.savefig('Error')
    plt.show()

def graficaPeso(pesos, valor_x):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots()
    ax.set_title('Gráfica evolución de los pesos')
    ax.set_xlabel('Iteración')
    ax.set_ylabel('Lista de pesos')
    ax.plot(numpy.arange(valor_x), pesos, label='Pesos')
    ax.legend(loc='best')
    plt.savefig('Pesos')
    plt.show()
   
def versus(y_deseada, y_calculada):
    valor_x = range(1,len(y_deseada)+1)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots()
    ax.set_title('Y deseada y Y calculada final')
    ax.plot(valor_x, y_deseada, color="red", marker="o", label="Y Deseada")
    ax.plot(valor_x, y_calculada, color="blue", marker="o", label="Y Calculada")
    ax.legend(loc="upper right")
    plt.savefig('Grafica Y')
    plt.show()

def err_versus(y_deseada, iteraciones, error_observado):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots()
    ax.set_title('ID Muestra vs Error observado')
    ax.set_ylabel('Error observado')
    ax.set_xlabel('Identificador de la muestra')
    x_values = np.arange(len(y_deseada))
    ax.plot(x_values, error_observado[0], marker='o', linestyle='solid', color="b", label="Error inicial")
    ax.plot(x_values, error_observado[int(iteraciones/2)-1], marker='o', linestyle='solid', color="g", label="Error medio")
    ax.plot(x_values, error_observado[iteraciones-1], marker='o', linestyle='solid', color="r", label="Error final")
    ax.legend()
    ax.set_xticks(x_values)
    ax.set_xticklabels([str(i+1) for i in range(100)])
    plt.xticks(range(0,100,9),rotation = 90)
    plt.grid()
    fig.tight_layout(pad=4.0)
    plt.show()
def reporte(pesos, error_permisible, magnitud_error, epocas, maximo_error):
    maximo_error.sort(reverse=True)
    titulos = ['Pesos iniciales', 'Pesos finales', 'Error permisible', 'Error observado', 'iteraciones', 'Max Error observado']
    with open('reporte.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=titulos)
        writer.writeheader()
        writer.writerow({
            'Pesos iniciales': pesos[0],
            'Pesos finales': pesos[-1],
            'Error permisible': round(error_permisible, 2),
            'Error observado': round(magnitud_error, 2),
            'iteraciones': epocas,
            'Max Error observado': round(maximo_error[0], 2)
        })
root = tk.Tk()
Ifz = Interfaz(root)
Ifz.interfaz()