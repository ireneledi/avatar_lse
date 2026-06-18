import pandas as pd
import numpy as np
import stanza

nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma')

class Word:
    def __init__(self, ident, text, lemma, upos, xpos, pos):
        self.id = ident
        self.text = text
        self.upper_text = str(text.upper())
        self.lemma = str(lemma.upper())
        self.upos = str(upos.lower()) if upos != None else ""
        self.xpos = str(xpos.lower()) if xpos != None else ""
        self.pos = pos


def prepareText(str_text, nlp):
    doc = nlp(str_text) ## Separa el texto en frases con palabras y sus caracteristicas
    text = []  ## Texto: listado de frases

    ## Iterar las frases extraidas
    for sent in doc.sentences:
        sentence = [] ## Frase: listado de palabras
        for word in sent.words:
            sentence.append(Word(word.id, word.text, word.lemma, word.upos, word.xpos, word.id-1))
        text.append(sentence) ## Insertar frase formateada en el texto

    return text


## Se comprueba si el valor de rule está en mayúscula y coincide con lemma
def isLemma(lemma, rule):
    return (rule.isupper() and (lemma == rule))


## Se comprueba si el valor de rule está en minúscula y coincide con upos o xpos
def isUPOS(upos, rule):
    return rule.islower() and (upos == rule)


def isXPOS(xpos, rule):
    if rule.islower() and (len(xpos) == len(rule)):
        arr_xpos = np.array(list(xpos))
        arr_rule = np.array(list(rule))

        zero_pos = np.where(arr_rule == '0') ## Posiciones con un 0
        arr_xpos[zero_pos] = '0' ## Sustituir posiciones por 0

        return np.array_equal(arr_xpos, arr_rule)

    return False


## Función que comprueba si una palabra coincide con la descripción de la regla
def checkPartialRule(word, rule):
    return (isLemma(word.lemma, rule) or isLemma(word.upper_text, rule) or isUPOS(word.upos, rule) or isXPOS(word.xpos, rule))


def checkRule(sentence, len_sentence, rule, current_pos_word=0, current_pos_rule=1, current_solution=[], solutions=[]):
    ## Si la longitud de la solución es la misma que la de la regla,
    ## se ha encontrado una solución completa
    if(len(current_solution) >= len(rule) and current_pos_rule > len(rule)):
        solutions.append(current_solution) ## Insertar la solución actual al listado de soluciones definitivas
        current_solution = [] ## Reiniciar la solucion actual
        current_pos_rule = 1 ## Reiniciar posicion

    ## Si la palabra actual es la última, FIN DE LA RECURSIVIDAD
    if(current_pos_word == len_sentence):
        return solutions ## Devolver las soluciones encontradas

    current_word = next((w for w in sentence if w.pos == current_pos_word),None)

    if len(current_solution) < (current_pos_rule):
        current_solution.append([])

    if(current_word == None):
        current_solution = []
        current_pos_rule = 1
    ## Comprobar si la palabra actual de la frase coincide con
    ## la descripción de la palabra actual de la regla
    elif(checkPartialRule(current_word, rule[current_pos_rule])):
        current_solution[current_pos_rule-1].append(current_word.id)
        current_pos_rule+=1
    elif rule[current_pos_rule] == '*':
        if(current_pos_rule+1 in rule.keys() and checkPartialRule(current_word, rule[current_pos_rule+1])):
            current_solution.append([current_word.id])
            current_pos_rule+=2
        else:
            current_solution[current_pos_rule-1].append(current_word.id)

        if(current_pos_word+1 == len_sentence):
            #Descartar la ultima parte de la regla si es *
            if(current_pos_rule == len(rule.keys()) and rule[current_pos_rule] == '*'):
                current_solution.append([])

            current_pos_rule+=1
    else:
        current_solution = []
        current_pos_rule = 1

    ## LLAMADA RECURSIVA: Comprobar si la siguiente palabra cumple
    return checkRule(sentence, len_sentence, rule, current_pos_word+1, current_pos_rule, current_solution, solutions)


def applyRules(sentence, rules, nlp):
    nwords = len(sentence) ## Numero de palabras en la frase
    pos = 1  ## Posición actual
    new_sentence = list(sentence) ## Frase transformada/actualizada
    exclusive_rules_applies = [] ## Reglas excluyentes aplicadas

    ## Iterar reglas de transformación
    for id_rule, row in rules.iterrows():

        ## Obtener la estructura de entrada y salida de la regla
        input_rule = ruleAsDictionary(row['input'])
        output_rule = ruleAsDictionary(row['output'])
        exclusive_rule = row['exclusive']


        ## Si la regla no es excluyente o no se ha aplicado otra que la excluya
        if exclusive_rule is None or not exclusive_rule in exclusive_rules_applies:
            ## Buscar las palabras de la frase que coinciden con la estructura de entrada
            new_sentence.sort(key=lambda x: x.pos)
            len_sentence = sum(word.pos>-1 for word in new_sentence)
            list_coincidences = checkRule(new_sentence, len_sentence, input_rule, 0, 1, [], [])

            if len(list_coincidences)>0:
                if not exclusive_rule is None:
                    exclusive_rules_applies.append(exclusive_rule) ## Insertar como aplicada

                for coincidence in list_coincidences:
#                    print("\n\nAPLICAR REGLA:")
#                    print("Input Rule: ", input_rule)
#                    print("Output Rule: ", output_rule)
#                    print("Coincidencia: ", coincidence)
                    coincidence_flatten = [j for sub in coincidence for j in sub]

                    ## Posicion del primer elemento que cumple con la regla
                    pos = next(word.pos for word in new_sentence if coincidence_flatten[0]==word.id)

                    ## Identificadores de las ultimas palabras que no cumplen con la regla
                    pos_last = next(word.pos for word in new_sentence if coincidence_flatten[-1]==word.id)
                    ids_last = [word.id for word in new_sentence if pos_last<word.pos]

                    ## Eliminar palabras del input rule que no estan en el output, poniendo posición a -1
                    ids_input_delete = list(iw for iw in list(input_rule) if iw not in list(output_rule))
                    for id_input_delete in ids_input_delete:
                        for id_delete in coincidence[id_input_delete-1]:
                          ## Busqueda de palabra con id a eliminar
                            for x in new_sentence:
                                if x.id == id_delete:
                                    x.pos = -1

                    ## Iterar las palabras en el output rule
                    for output_id in output_rule:
                        ## Añadir una nueva palabra, si el id en el output es 0 (no está en el input)
                        if output_id == 0:
                            nwords += 1
                            new_word_nlp = nlp(output_rule[output_id]).sentences[0].words[0]
                            new_word = Word(nwords, output_rule[output_id], output_rule[output_id], new_word_nlp.upos, new_word_nlp.xpos, pos)
                            new_sentence.append(new_word) ## Insertar palabra
                            pos += 1 ## Incrementar posición actual

                        ## Actualizar las palabras del output que estan en el input
                        elif output_id in list(input_rule) :

                            ## Descripción la palabra actual del input y output rule
                            input_word = input_rule[output_id]
                            output_word = output_rule[output_id]

                            for id_word in coincidence[output_id-1]:
                                ## Actualizar la posición de la palabra con la posición actual
                                current_word = next(x for x in new_sentence if x.id == id_word)
                                current_word.pos = pos
                                pos += 1 ## Incrementar posición actual

                                ## Si la descripcion del input y el output no coinciden
                                if(input_word != output_word):
                                    replace_word = output_word.replace(input_word, current_word.lemma.upper())
                                    current_word.lemma = replace_word   ## Modificar lema de la palabra

                    #Actualizar posiciones del resto de palabras de la frase
                    for word in new_sentence:
                        if word.id in ids_last:
                            word.pos = pos
                            pos += 1 ## Incrementar posición actual
                    new_sentence.sort(key=lambda x: x.pos)

                ## Imprimir la frase transformada tras aplicar una regla
#                print(glossSentece(new_sentence))

    return new_sentence


## Función que transforma el formato de una regla: de string a diccionario
## Las claves del diccionario sera la posición de las palabras (enteros)
## Los valores del diccionario sera la representación de las palabras (string)
def ruleAsDictionary(rule):
    dic_rule = {}
    ## Si la regla es nula, se devuelve un dict vacío
    if rule!=rule:
        return {}

    formatted_rule = rule.strip() ## Eliminar espacios del principio y fin
    formatted_rule = " ".join(formatted_rule.split()) ## Reemplazar multiples espacios por uno

    ## Separar el string por los delimitadores indicados
    list_rule = formatted_rule.split(" ")
    for elem in list_rule:
        ident, desc = elem.split("_", 1)
        dic_rule[int(ident)] = desc

    return dic_rule


def glossSentece(sentence):
    result = ""
    ## Iterar posiciones de la frase
    for pos in range(0, len(sentence)):
        current_word = next( (w for w in sentence if w.pos == pos), None) ## Busqueda de palabra en la posicion actual
        if current_word != None:
            result += current_word.lemma + " " ## Lema de la palabra en la posicion actual
    return result


def insertCorpus(path_corpus, input_sentence, gloss_sentence):
    df = pd.DataFrame({'sentence': [str(input_sentence)], 'gloss': [str(gloss_sentence)]})
    df.to_csv(path_corpus, mode='a', index=False, header=False)


def generatorText2Gloss(input_text, path_rules, path_corpus, nlp, max_len = 50):
        text = prepareText(input_text, nlp) ## Listado de frases

        rules = pd.read_csv(path_rules, encoding='utf-8') ## Lectura de reglas

        final_gloss = ""

        for sentence in text:
                if sentence != None:
                        nwords = len(sentence)
                        if nwords <= max_len:
                                input_sentence = ' '.join([word.text for word in sentence]) ## Frase original

                                ## Generar la glosa de la frase
                        #        print("FRASE ORIGINAL: ", input_sentence)
                                new_sentence = applyRules(sentence, rules, nlp) ## Aplicación de reglas
                                gloss_sentence = glossSentece(new_sentence) ## Frase glosada
                                final_gloss += gloss_sentence + " " # Obtener las glosas de más de una frase
                        #        print("\n\nRESULTADO GLOSA: ", gloss_sentence)

                                ## Almacenamiento del resultado...
                                insertCorpus(path_corpus, input_sentence, gloss_sentence)

        return final_gloss.strip()
