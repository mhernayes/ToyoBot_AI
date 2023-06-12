import gspread as gs
import pandas as pd

gc = gs.service_account(filename='toyobot-295848405f06.json')

def readContacts(fileName):
    lst = []
    
    file =  gc.open(fileName)
    sheet = file.worksheet('base')#.range('B3:W')
    # Trae la data como diccionarios y transformarlos a un dataframe
    listaDicts = sheet.get_all_records()
    listaKeys = list(listaDicts[0].keys())
    listaValuesd = [ list(dicts.values()) for dicts in listaDicts]

    listaValueso = {}
    for i in range(len(listaKeys)):
        valores = []
        for values in listaValuesd:
            valores.append(values[i])
            listaValueso[listaKeys[i]]= valores[:]
    
    lst = listaValueso

    
    return lst

nros = readContacts('Base de Datos')

