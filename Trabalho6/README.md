# Projeto Demonstrativo 6 - Rastreamento de objetos

## Organização dos arquivos do projeto
Para manter um certo padrão, foi criada uma pasta "data" pasta servir de diretório para todos os datasets com a seguinte organização:
Data/Base_de_dados/Descricao_dos_dados/imagens e groundTruth (ex: data/Professor/car1)

Todos os outros datasets utilizados foram baixados do link recomendado pelo professor: http://cmp.felk.cvut.cz/~vojirtom/dataset/tv77/

No entanto, para facilitar a execução e correção, um espelho da pasta de dados usada para fazer o trabalho está em: https://drive.google.com/open?id=1dAxbXiot7lW6CcGXN6CgmwjU_vnguOWr

Basta baixá-la e colocá-la na pasta raiz junto aos arquivos do projeto.

## Execução (algoritmos OpenCV)
Para rodar o programa com os algoritmos implementados no OpenCV, basta executar o comando:
    python track.py

Por padrão, o programa vai executar sem nenhum filtro, no dataset car1 e com o tracker BOOSTING. Para executar utilizando outro tracker, basta executar do seguinte modo:
    python track.py -t [TrackerName]

Onde o TrackerName pode ser: BOOSTING, MIL, KCF, TLD ou MEDIANFLOW

Para executar utilizando outro dataset, basta executar do seguinte mmodo:
    python track.py -d [DatasetName]

Onde DatasetName = Base_de_dados + _ + Descrica_dos_dados

Por exemplo, "Professor_car1" é um dos datasets passados pelo professor, "Babenko_girl" é o da base Babenko no video girl.

## Execução (template matcher, filtro de partículas)
Para rodar o programa com o algoritmo de template matching como rastreador, basta executar o comando:
    python track_alt.py

Por padrão, o algoritmo será executa sem filtro de partículas, no dataset car1 e com matcher baseado no quadrado das diferenças. Para executar em outro dataset, basta usar o mesmo comando dado acima:
    python track_alt.py -d [DatasetName]

Para executar com filtro de partículas, basta usar a chave '-p':
    python track_alt.py -p

Caso queria se alterar o número de partículas ou matcher utilizado, pode-se usar variações do comando:
    python track_alt.py -p -n [NumParticles] -m [Matcher]

Em que NumParticles é um número inteiro e Matcher pode ser 'ccoeff', 'ccorr' e 'sqdiff'.