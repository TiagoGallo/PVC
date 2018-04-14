import cv2
import numpy as np

class Req1:
    def __init__(self):
        '''
            Construtor da classe que cumpre o requisito 1 do segundo trabalho de PVC (1/2018)
            Cria uma janela do openCV com o nome "Requisito 1", determina o método de callback
        clique do mouse e inicia algumas variáveis que serâo utilizadas no programa.
        '''
        cv2.namedWindow("Requisito 1")
        cv2.setMouseCallback('Requisito 1',self.get_mouse_position)

        #inicializa coordenadas iniciais e finais e o numero de cliques que ja ocorreram
        self.xi = None
        self.yi = None
        self.xf = None
        self.yf = None
        self.num_cliques = 0

        #instancia objeto da webcam
        self.webcam = cv2.VideoCapture(0)

    def get_mouse_position(self, event,x,y,flags,param):
        '''
            Método de callback para clique do mouse, quando ocorrer um clique do mouse na janela
        "Requisito 1" do OpenCV esse método será chamado e vai salvar as coordenadas do mouse na
        tela. A primeira vez que o método for chamado ele vai salvar as coordenadas iniciais do
        objeto, a segunda vez ele salva as coordenadas finais e em todas as outras vezes ele não 
        faz nada.
        '''
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.num_cliques == 0:
                self.xi = x
                self.yi = y
                self.num_cliques += 1
                print("Primeiro clique na coordenada ({},{})".format(x,y))

            elif self.num_cliques == 1:
                self.xf = x
                self.yf = y
                self.num_cliques += 1
                print("Segundo clique na coordenada ({},{})".format(x,y))
                self.calc_euclidian_distance()

            else:
                print("Os dois pontos ja foram pegos e a distancia calculada, execute o programa "
                    "novamente para poder escolher dois novos pontos")
                pass

    def calc_euclidian_distance(self):
        '''
            Método para calcular a distância eucliadiana bidimensional entre dois pontos da imagem,
        será chamado quando clicarmos no segundo ponto da tela e vai imprimir o resultado no console.
        '''
        dx = (self.xi - self.xf) ** 2
        dy = (self.yi - self.yf) ** 2

        dist = np.sqrt(dx + dy)

        print("A distancia euclidiana dos pontos foi {}".format(dist))

    def run(self):
        '''
            Método principal do programa, fica em um loop mostrando a imagem da webcam e esperando
        o usuário clicar nos pontos desejados. Após o usuário ter clicado nos dois pontos desejados
        o loop fica desenhando uma reta entre esses dois pontos sempre que vai mostrar a imagem da 
        webcam.
        '''
        while True:
            #Tenta ler o próximo frame da webcam e se não for lido, sai do programa
            grab, imagem = self.webcam.read()
            if not grab:
                print("Erro ao ler imagem da Webcam")
                break

            #Se já tiver pego as duas coordenadas desejadas, desenha a linha na imagem
            if self.num_cliques > 1:
                cv2.line(imagem,(self.xi,self.yi),(self.xf,self.yf),(255,0,0),5)

            cv2.imshow("Requisito 1", imagem)

            #Aperte a tecla 'q' para encerrar o programa
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break

if __name__ == "__main__":
    Req1().run()