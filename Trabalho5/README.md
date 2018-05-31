Para instalar todos os pacotes necessários para rodar os scripts, basta executar

pip install -r requirements.txt

Primeiramente, para conseguir replicar os experimentos que foram feitos, é necessárrio baixar os pesos das redes que nós treinamos. Esses pesos podem ser baixados no seguinte link:

https://drive.google.com/drive/folders/1OVIaZ8b90YdV9FrzLBFlfjBpJIVTGJ6v?usp=sharing

Os scripts utils.py e datasetHelper.py são apenas scripts  de apoio para a nossa finalidade.
Os scrips que realmente são utilizados são o que faz o treino, o que calcula acuracia e o que mostra os resultados em algumas imagens 

Para treinar uma nova rede, basta rodar o scrpit train.py com o seguinte comando:

python train.py -d path/to/training/dataset 

os pesos da rede treinada serão salvos em ./weights/InceptionV3.h5 (portanto, essa pasta weights deve ser criada antes de rodar o script)

Para calcular a acurácia de um modelo basta executar:

python evaluateNetwork.py -d path/to/test/dataset -m path/to/model/weights

Para ver o resultado da rede em algumas imagens basta executar 

python seeResults.py -d path/to/test/images -m path/to/model/weights