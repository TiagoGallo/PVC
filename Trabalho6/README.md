Para executar o programa basta executar:

python track.py

Por padrao, o programa vai executar sem Kalman Filter, no dataset car1 que o professor passou e com o tracker BOOSTING.

Para executar com Kalman Filter, basta executar da seguinte maneira:

python track.py -k 1

Para executar utilizando outro tracker, basta executar do seguinte modo:

python track.py -t [TrackerName]

onde o TrackerName pode ser: BOOSTING, MIL, KCF, TLD ou MEDIANFLOW

Para executar utilizando outro dataset, basta executar do seguinte mmodo:

python track.py -d [DatasetName]

onde, DatasetName = Base_de_dados + _ + Descrica_dos_dados
Por exemplo, "Professor_car1" eh um dos datasets passados pelo professor, "Babenko_girl" eh o da base Babenko no video girl.

Para manter um certo padrao, foi criada uma pasta "data" no mesmo diretorio dos arquivos que contem todos os datasets com a seguinte organizacao:
Data/Base_de_dados/Descricao_dos_dados/imagens e groundTruth (ex: data/Professor/car1)

Todos os outros Datasets utilizados foram baixados do link recomendado pelo professor: http://cmp.felk.cvut.cz/~vojirtom/dataset/tv77/
No entanto, para facilitar a correcao, a pasta de dados exata que usei para fazer o trabalho esta upada em: 