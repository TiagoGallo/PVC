Para preparar o ambiente com todas as bibliotecas necessárias para rodar os programas, basta executar no terminal:

pip install requirements.txt

Os métodos com base nos espaços cromáticos contém um arquivo .sh para ajudar na execução, basta alterar os paths para o dataset que estão nesse arquivo e executar ele com:

./color_space.sh

Que é possível adquirir os mesmo resultados que nós.

Para executar o script que implementa KNN, basta alterar os paths para o dataset nas linhas 47 a 52, e os paths para imagem e mascara de teste nas linhas 124 e 125 e executar:

python knn.py

Para executar o scrip da unet, voce precisa clonar o repositorio https://github.com/TiagoGallo/PVC/tree/master/Trabalho4 e executar:

python unet.py

Com isso voce conseguira reproduzir os resultados apresentados no relatorio.
Para treinar novamente a Unet, basta colocar como True a variavel train_mode da linha 152 e executar:

python unet.py

Os novos pesos serao salvos com o nome peso3.h5 e quando voce desativar o train_mode e rodar o codigo de novo, esses novos pesos ja seram utilizados para inferir as imagens