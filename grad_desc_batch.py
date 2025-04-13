import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing as dados

def f(x, a, b):
    '''
    ENTRADAS:
        float x: Abcissas usadas para calcular

    FUNCIONAMENTO: 
        O método calcula uma função qualquer adotada
        (nesse caso, f(x) = ax + b), no determinado ponto x fornecido.

    SAÍDA:
        float y: Ordenada de f(x) 
    '''
    return a*x + b

def der_parc_a(X_REAL, Y_REAL, Y_CALC):
    '''
    ENTRADAS:
        float[] X_REAL: Array de x's reais
        float[] Y_REAL: Array de y's reais
        float[] Y_CALC: Array de y's calculados em f(x)

    FUNCIONAMENTO: 
        A função realiza o cálculo da Derivada Parcial com
        Relação a textit{a}.

    SAÍDA:
        float der_parc_a: Valor da derivada parcial com relação a textit{a}. 
    '''

    n = len(X_REAL)

    sigma = sum(x * (y - y_obs) for x,y,y_obs in zip(X_REAL,Y_REAL,Y_CALC))
    
    return (2/n) * sigma

def der_parc_b(Y_REAL, Y_CALC):
    '''
    ENTRADAS:
        float[] X_hats: Array de x's observados
        float[] Ys: Array de y's reais
        float[] Y_obs: Array de y's observados

    FUNCIONAMENTO: 
        A função deve realizar o cálculo da Derivada Parcial com
        Relação a b.

    SAÍDA:
        float der_parc_b: Valor da derivada parcial com relação a b. 
    '''
    n = len(X_REAL)

    sigma = sum(y - y_obs for y,y_obs in zip(Y_REAL,Y_CALC))

    return (2/n) * sigma
    
def eqm(Y_REAL, Y_CALC):
    '''
    ENTRADA:
        float[] Y_REAL: array de ordenadas observadas na amostra        
        float[] Y_CALC: array de ordenadas calculadas por f(x_real)
    
    FUNCIONAMENTO:
        A função calcula o erro quadrático médio entre o 
        valor observado na amostra e o valor calculado por f(x)

    SAÍDA:
        float eqm:  valor do erro quadrático médio para o conjunto Y_CALC
    '''

    n = len(Y_REAL)
    
    eqm = sum((1/n)*(y_real - y_calc)**2 for y_real, y_calc in zip(Y_REAL,Y_CALC))

    return eqm

def grad_desc_iter(a, b, alpha, epsilon, epocas, X_REAL, Y_REAL, Y_CALC):
    '''
    ENTRADAS:
        float   a:        coeficiente angular de f(x)
        float   b:        coeficiente linear de f(x)
        float   alpha:    learning rate do método
        float   epsilon:  erro tolerado
        int     epocas:     quantidade de iteração

    FUNCIONAMENTO: 
        A função realiza o método de minimização do 
        gradiente descendente de modo iterativo.

    SAÍDA:
        float[] DADOS_FINAIS:   array com os valores da abcissa xk1 e dos
                                coeficientes textit{a} e textit{b} quando |xk1 - xk| <= epsilon.
    '''

    EQMs = []

    xk1 = f(a, b)
    xk = 0

    ## LOOP DO GRADIENTE DESCENDENTE ##
    for _ in range(epocas):

        ### condição de parada ###
        if eqm(Y_REAL, Y_CALC) <= epsilon:
                    print(f"A função minimizada é aproximadamente f(x) = {a}x + {b} com epsilon = {epsilon}")

                    DADOS_FINAIS = [xk1, a, b]

                    return DADOS_FINAIS
        
        
        ### valores de f nas abcissas da amostra
        Y_CALC = [f(x, a, b) for x in X_REAL]


        ### DERIVADAS PARCIAIS ###
        dpa = der_parc_a(X_REAL, Y_REAL, Y_CALC)
        dpb = der_parc_b(Y_REAL, Y_CALC)
    
        
        xk = xk1

        a -= alpha*dpa
        b -= alpha*dpb
        
        xk1 = f(xk, a, b)

        EQMs.append(eqm(Y_REAL, Y_CALC))

        


def __main__():
    epocas = int(sys.argv[1])       # numero de iteracoes
    alpha = float(sys.argv[2])      # learning rate
    epsilon = float(sys.argv[3])    # erro tolerado

    
    ### dados do california housing ###
    X_REAL = dados().data[:, 0]     # dados de entrada (features)
    Y_REAL = dados.target           # variável alvo (preço médio de casa)
    

    # coeficientes de f gerados aleatoriamente
    a = random.uniform()            # coeficiente angular de f
    b = random.uniform()            # coeficiente linear de f
    
    # valores de f nas abcissas dos pontos
    Y_CALC = [f(x, a, b) for x in X_REAL]

    grad_desc_iter(a, b, alpha, epsilon, epocas, X_REAL, Y_REAL, Y_CALC)
    

    '''
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color='blue', label='Dados Originais')
 
    ## PLOTAGEM ##
    plt.plot(X, Y_obs, linestyle='-', color='r', linewidth=2, label="Reta minimizada")

    plt.title("Gradiente Descendente Iterativo")
    plt.xlabel("Eixo x")
    plt.ylabel("Eixo y")
    plt.show()'''

if __name__ == "__main__":
    __main__()
