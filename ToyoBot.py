#importamos la libreria de openai
import openai  # pip install openai
import typer  # pip install "typer[all]". libreria donde puedo agregar prompts y cosas copadas
from rich import print  # pip install rich. libreria para hacer mas bonitos los prints
from rich.table import Table
#importamos el archivo config donde vamos a guardar la api key
import config
import wppbot
import gspread as gs
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


"""
Webs de interés:
- Módulo OpenAI: https://github.com/openai/openai-python
- Documentación API ChatGPT: https://platform.openai.com/docs/api-reference/chat
- Typer: https://typer.tiangolo.com
- Rich: https://rich.readthedocs.io/en/stable/
"""

#nos conectamos al archivo clase.json donde estan guardadas todas las privates key
gc = gs.service_account(filename = "toyobot-295848405f06.json")

#definimos la funcion main donde ingresaron el codigo principal del chat. esta funcion es la que definimos en typer

import re

def preprocesar_texto(texto):
    # Convertir a minúsculas
    texto = texto.lower()

    # Tokenización
    palabras = re.findall(r'\b\w+\b', texto)

    # Eliminar caracteres no deseados
    palabras_limpias = [palabra for palabra in palabras if len(palabra) > 2]

    # Unir las palabras en una sola cadena de texto
    texto_limpiado = ' '.join(palabras_limpias)

    return texto_limpiado


def main():
    #llamamos a la api key desde otro archivo llamado config donde esta guardada con la variable api_key
    openai.api_key = config.api_key

    #definimos la varialbe table como una Tabla y le agregamos 2 filas "exit" y "new" 
    table_1 = Table("💬 [bold red]Asistente de Toyota[/bold red]")
    table_1.add_row("Realizado con versión de ChatGTP 3.5 Turbo")
    table_2 = Table("Menu", "Descripción")
    table_2.add_row("exit", "Ingrese exit SALIR de la aplicación")
    table_2.add_row("new", "Ingrese new par CREAR una nueva conversación")
    
    #imprimimos la tabla
    print(table_1)
    print(table_2)

    #utilizamos esta base de datos creada en Sheet para levantar informacion y asi entrenar a la AI
    datos = wppbot.readContacts("Base de Datos")
    
    pregunta = datos["Pregunta"]
    respuesta = datos["Respuesta"]

    # Preprocesamiento de datos
    pregunta_procesada = [preprocesar_texto(preg) for preg in pregunta]
    respuesta_procesada = [preprocesar_texto(resp) for resp in respuesta]

    # Concatena las preguntas y respuestas preprocesadas
    texto = '\n'.join([f'Q: {preg}\nA: {resp}' for preg, resp in zip(pregunta_procesada, respuesta_procesada)])

    # Concatena todas las preguntas y respuestas en un solo texto
    texto = '\n'.join([f'Q: {pregunta}\nA: {respuesta}' for pregunta, respuesta in zip(pregunta, respuesta)])
    
    # Crear una instancia del vectorizador TF-IDF
    vectorizador = TfidfVectorizer()

    # Calcular la matriz TF-IDF
    matriz_tfidf = vectorizador.fit_transform([texto])

    role_content = "Durante la conversacion solo podeés contestar con la informacion que este en la base de datos "+ texto + "y si la informacion no está en la base de datos decí: 'Lo siento, no cuento con esa información'"
    context = {"role": "system",
               "content": role_content}
    #guardamos el contexto del asistente en una lista llamada messages
    messages = [context]
    print("Buenos días, soy un Asistente de Toyota.")

    #creamos un bucle para que siga preguntandonos cosas
    while True:
        #llamamos a la funcion __prompt donde emepzamos a preguntar 
        content = __prompt()

        #si quiseramos crear una conversacion nueva volvemos a limpiar la lista messages en "context"
        if content == "new":
            print("🆕 Nueva conversación creada")
            messages = [context]
            #llamamos a la funcion content para que pregunte nuevamente que quiere preguntar
            content = __prompt()

        #vamos añadiendo a la lista message un json con el "role" y "content" que es el contenido de los que vamos preguntando
        messages.append({"role": "user", "content": content})

        #guardamos la respuesta en una variable que se llama response y le pasamos 2 parametros
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", #1 el modelo a utilizar: gpt-3.5-turbo
                                                temperature = 0.0, #definimos la temperatura en 0 para que las respuestas sean lo mas especificas posibles
                                                messages=messages) #2 el mensaje: la lista con todos el "content" que vamos añadiendo

        #nos quedamos con el contenido de la respusta ya que content es una lista con mas informacion
        response_content = response.choices[0].message.content
        
        #vamos añadiendo a la lista messages "response_content" que es el contenido de lo va respondiendo
        messages.append({"role": "assistant", "content": response_content})

        # Calcular el vector TF-IDF de la consulta
        vector_tfidf_consulta = vectorizador.transform([content])

        # Calcular la similitud entre la consulta y las preguntas almacenadas
        similitudes = cosine_similarity(vector_tfidf_consulta, matriz_tfidf).flatten()

        # Obtener el índice de la pregunta más similar
        indice_pregunta_similar = similitudes.argmax()

        # Obtener la pregunta más similar y su respuesta correspondiente
        pregunta_similar = pregunta[indice_pregunta_similar]
        respuesta_similar = respuesta[indice_pregunta_similar]

        # Añadir la respuesta similar a la lista messages
        messages.append({"role": "assistant", "content": respuesta_similar})


        print(f"[bold blue]> [/bold blue] [blue]{response_content}[/blue]")


#definomos la funccion __prompt para preguntar que queiere hacer el usuario 
def __prompt():
    prompt = typer.prompt("\n¿En qué te puedo ayudar?")

    #si el usuario ingresa salir 
    if prompt == "exit":
        exit = typer.confirm("😨 ¿Estás seguro?")
        if exit:
            print("👋¡Hasta luego!👋")
            #detener la ejecucion de typer
            raise typer.Abort()
        #si no quiero salir vuelvo a llamar a la funcion __prompt
        return __prompt()

    return prompt

#necesitamos este "if" para correr la libreria typer y main que es la funcion principal
if __name__ == "__main__":
    typer.run(main)