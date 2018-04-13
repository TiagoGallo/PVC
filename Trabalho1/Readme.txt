Objetivo
Esta atividade tem como objetivo principal a exploração e desenvolvimento de algoritmos na ferramenta OpenCV (http://opencv.org/).
Procedimentos
Observação: Todos os requisitos desta atividade deverão ser elaborados utilizando a versão 2.x (ou superior) do ferramenta OpenCV.

Requisito 1: Elabore uma aplicação utilizando OpenCV que abra um arquivo de imagem (tipo JPG) e que permita ao usuário clicar (botão esquerdo do mouse) sobre um ponto na área da imagem na tela e após realizado o clique mostre no terminal a coordenada do ponto (x,y) na imagem, informando os valores do pixel RGB, quando a imagem for colorida ou mostrando o valor da intensidade do pixel quando a imagem for em nível de cinza (grayscale).

Requisito 2: Repita o procedimento desenvolvido no Requisito 1 e crie uma rotina de seleção de pixels baseado na cor de onde for clicado. Seu programa deve comparar o valor da cor (ou tom de cinza) de todos os pixels da imagem com o valor da cor (ou tom de cinza) de onde foi clicado. Se a diferença entre esses valores for menor que 13 tons marque o pixel com a cor vermelha e exiba o resultado na tela. 

Requisito 3: Repita o procedimento desenvolvido no Requisito 2, em que ao invés de abrir uma imagem, abra um arquivo de vídeo (padrão avi ou x264) e realize os mesmos procedimentos do Requisito 2 durante toda a execução do vídeo.

Requisito 4: Repita o procedimento desenvolvido no Requisito 3, em que ao invés de abrir um arquivo de vídeo, a aplicação abra o streaming de vídeo de uma webcam ou câmera USB conectada ao computador e realize todos os procedimentos solicitados no requisito 3.
-----------------------------------------------------------------------------------------------

Instruções para Elaboração do Relatório
O relatório deve demonstrar que a respectiva atividade de laboratório foi realizada com sucesso e que os princípios subjacentes foram compreendidos.


O relatório da atividade de projeto demonstrativo é o documento gerado a partir do trabalho realizado seguindo as orientações exigidas na metodologia para se atender ao requisito solicitado. Este deve espelhar todo o trabalho desenvolvido nos processos de obtenção dos dados e sua análise. Apresentamos a seguir uma recomendação de organização para o relatório da atividade de laboratório. Deverá conter as seguintes partes:


i. Identificação: Possuir a indicação clara do título do projeto demonstrativo abordado, a data da sua realização, a identificação da disciplina/turma, nome do autor, e quando houver, número de matrícula e email.

ii. Objetivos: Apresentar de forma clara, porém sucinta, os objetivos do projeto demonstrativo.

iii. Introdução: Deve conter a teoria necessária à realização da atividade do projeto demonstrativo proposto. Utilize sempre fontes bibliográficas confiáveis (livros e artigos científicos), evitando utilizar única e exclusivamente fontes de baixa confiabilidade (Wikipedia, Stackoverflow,...).

iv. Materiais e Metodologia empregada: É dedicada à apresentação dos materiais e equipamentos, descrição do arranjo experimental e uma exposição minuciosa do procedimento do projeto demonstrativo realmente adotado.

v. Resultados: Nesta parte são apresentados os resultados das implementações efetuadas, na forma de tabelas e gráficos, sem que se esqueça de identificar em cada caso os parâmetros utilizados.

vi. Discussão e Conclusões: A discussão visa comparar os resultados obtidos e os previstos pela teoria. Deve se justificar eventuais discrepâncias observadas. As conclusões resumem a atividade de laboratório e destacam os principais resultados e aplicações dos conceitos vistos.

vii. Bibliografia: Citar as fontes consultadas, respeitando as regras de apresentação de bibliografia (autor, título, editora, edição, ano, página de início e fim).

O relatório do laboratório deverá ser confeccionado em editor eletrônico de textos com no máximo 4 (quatro) páginas, utilizando obrigatoriamente o padrão de formatação descrito no arquivo de exemplo disponibilizado aqui. Está disponibilizado um padrão para editores científicos LATEX (arquivo extensão *.zip contendo arquivo de exemplo do uso do pacote), cabendo ao aluno a escolha de qual editor/IDE será utilizado. Não serão permitidos relatórios confeccionados em outro editor eletrônico de texto, ou usando um modelo diferente do padrão LaTeX disponibilizado.

Instruções para Submissão da atividade de Projeto Demonstrativo
Esta tarefa consiste na submissão de um arquivo único Zipado, contendo um arquivo PDF do relatório elaborado e também o código fonte desenvolvido, obrigatoriamente em C/C++ ou Python, e um arquivo com diretivas de compilação em Linux. 

O código pode ser desenvolvido em ambiente Windows, mas o código submetido deverá ser obrigatoriamente compilável (para permitir a avaliação pelo professor/tutores) em ambiente Linux. Para referência, o ambiente Linux que será utilizado para a correção das atividades é Ubuntu 16.04 64 bits utilizando a versão 3.2.0 do OpenCV e Python 3.5.2. Reforçando que esta atividade é INDIVIDUAL.
