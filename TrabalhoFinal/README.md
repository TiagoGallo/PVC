# Projeto Final - Mouse adaptado

## Instalação dos pré-requisitos
Para instalar os pré-requisitos necessários para executar esses códigos, basta instalar as seguintes bibliotecas:

* Opencv
* Face_recognition
* PyAutoGui

Todas essas bibliotecas podem ser instaladas com um simples comando:

pip install {lib}

## Execução 
Para rodar o programa basta executar:

python main.py -d {DELAY_TIME} -c {CLICK_MODE} -m {MAX_ACCELARATION}

Todos os parâmetros são opcionais, portanto o programa pode ser executado sem nenhum deles ou com uma combinação deles.

### Descrição dos parâmetros

* DELAY_TIME (-d): É o tempo (em segundos) que o usuário precisa ficar sem mover o mouse ou com um olho fechado para caracterizar o clique do mouse. O valor default é 3.
* CLICK_MODE (-c): É o modo utilizado para fazer o clique do mouse, pode ser definido como 'dwell' (mouse será clicado quando parado) ou 'eye' (mouse será clicado quando o olho estiver fechado). O valor default é 'eye'.
* MAX_ACCELARATION (-m): Quando o mouse começa a se mover ele vai movimentando lentamente e acelerando conforme o tempo, esse parâmetro indica o valor máximo que o movimento do mouse pode atingir. Os valor significa quantos pixels o mouse vai ser movimentar em cada iteração. O valor default é 20.