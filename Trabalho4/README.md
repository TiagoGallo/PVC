Para preparar o ambiente com todas as bibliotecas necessárias para rodar os programas, basta executar no terminal:

pip install requirements.txt

Os métodos com base nos espaços cromáticos contém um arquivo .sh para ajudar na execução, basta alterar os paths para o dataset que estão nesse arquivo e executar ele com:

./color_space.sh

Que é possível adquirir os mesmo resultados que nós.

Para executar o script que implementa KNN, basta alterar os paths para o dataset nas linhas 47 a 52, e os paths para imagem e mascara de teste nas linhas 124 e 125 e executar:

python knn.py

