import  sys
import  random
import  matplotlib.pyplot as plt
from    sklearn.datasets import fetch_california_housing as dados



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
    
    return -(2/n) * sigma



def der_parc_b(Y_REAL, Y_CALC):
    '''
    ENTRADAS:
        float[] Y_REAL: Array de y's reais
        float[] Y_CALC: Array de y's calculados por f(x)

    FUNCIONAMENTO: 
        A função deve realizar o cálculo da Derivada Parcial com
        Relação a b.

    SAÍDA:
        float der_parc_b: Valor da derivada parcial com relação a b. 
    '''
    n = len(Y_REAL)

    sigma = sum(y_real - y_calc for y_real,y_calc in zip(Y_REAL,Y_CALC))

    return -(2/n) * sigma
    


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



def plota_pontos(DOMINIO, IMAGEM):

    plt.scatter(DOMINIO, IMAGEM, color='blue', label='Pontos da Amostra')

    return



def plota_reta(DOMINIO, IMAGEM, cor, label_reta, titulo, eixo_x, eixo_y):
 
    plt.plot(DOMINIO, IMAGEM, linestyle='-', color=cor, linewidth=2, label=label_reta)

    plt.title(titulo)
    plt.xlabel(eixo_x)
    plt.ylabel(eixo_y)
    plt.legend()

    return


def grad_desc_iter(a, b, alpha, epsilon, epocas, X_REAL, Y_REAL):
    '''
    ENTRADAS:
        float   a:          coeficiente angular de f(x)
        float   b:          coeficiente linear de f(x)
        float   alpha:      learning rate do método
        float   epsilon:    erro tolerado
        int     epocas:     quantidade de iterações
        float[] X_REAL:     array de abcissas da amostra
        float[] Y_REAL:     array de ordenadas da amostra

    FUNCIONAMENTO: 
        A função realiza o método de minimização do 
        gradiente descendente de modo iterativo.

    SAÍDA:
        float[] DADOS_FINAIS:   array com os valores dos coeficientes textit{a} e 
                                textit{b} quando |eqm_{k+1} - eqm_{k}| <= epsilon.
    '''

    EQMs = []                   # array de eqm's a cada iteração

    eqm_ant = 0.0               # erro quadrático médio anterior, usado para comparação


    ## LOOP DO GRADIENTE DESCENDENTE ##
    for epoca in range(epocas):

        ### valores de f nas abcissas da amostra
        Y_CALC = [f(x, a, b) for x in X_REAL]


        ### cálculo do eqm e adição ao array
        eqm_atual = eqm(Y_REAL, Y_CALC)
        EQMs.append(eqm_atual)
        

        ### CONDIÇÃO DE PARADA ###
        if abs(eqm_atual - eqm_ant) <= epsilon:
            
            print(f"A função minimizada ÓTIMA é aproximadamente f(x) = {a}x + {b} com epsilon = {epsilon}\n\n")
            
            titulo = f"Reta de f(x) ~ {a:.3f}x + {b:.3f} com {epocas} épocas \nlearning rate = {alpha}; epsilon = {epsilon}"
            label = "Função f(x) otimizada"

            plota_pontos(X_REAL,Y_REAL)
            plota_reta(X_REAL, Y_CALC, 'r', label, titulo, "Eixo x", "Eixo y")

            plt.show(block=False)

            return EQMs
        
        
        eqm_ant = eqm_atual     # atualização do eqm anterior


        ### DERIVADAS PARCIAIS ###
        dpa = der_parc_a(X_REAL, Y_REAL, Y_CALC)
        dpb = der_parc_b(Y_REAL, Y_CALC)


        ### ATUALIZAÇÃO DOS PARÂMETROS ###
        a -= alpha*dpa
        b -= alpha*dpb


        ### ÚLTIMA ITERAÇÃO ###
        if epoca == epocas - 1:
            
            print(f"A função parcialmente minimizada é f(x) ~ {a}x + {b} com epsilon = {epsilon}\n\n")

            titulo = f"Reta de f(x) ~ {a:.3f}x + {b:.3f} com {epocas} épocas \nlearning rate = {alpha}; epsilon = {epsilon}"
            label = "Função f(x) parcialmente minimizada"

            plota_pontos(X_REAL,Y_REAL)
            plota_reta(X_REAL, Y_CALC, 'r', label, titulo, "Eixo x", "Eixo y")
            
            plt.show()

            return EQMs



def main():
    epocas = int(sys.argv[1])       # numero de iteracoes
    alpha = float(sys.argv[2])      # learning rate
    epsilon = float(sys.argv[3])    # erro tolerado

    
    ### dados do california housing ###
    X_REAL = dados().data[:, 0]     # dados de entrada [0,5]
    Y_REAL = dados().target         # variável alvo (preço médio de casa)


    ### coeficientes de f gerados aleatoriamente ###
    a = random.uniform(-100,100)
    b = random.uniform(-100,100)    

    ERROS = grad_desc_iter(a, b, alpha, epsilon, epocas, X_REAL, Y_REAL)

    titulo_erro =   "Progressão do Erro Quadrático Médio (EQM) de f(x)"
    label_erro  =   "EQM"

    DOMINIO_ERRO = list(range(epocas))

    plota_reta(DOMINIO_ERRO, ERROS, 'r', label_erro, titulo_erro, "Iterações", "Erro")

    plt.show()


if __name__ == "__main__":
    main()
