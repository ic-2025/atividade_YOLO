# Monitoramento de Idosos usando Detecção de Objetos com YOLO e OpenCV

### Integrantes:
- Brendo dos Santos Carvalho 	(202106840027)\
- Joseph Frank Kwamina Abiw 	(201806840004)\
- Prince Nyarko 			(202006840047)


## Objetivos
Desenvolver um sistema de monitoramento para idosos que detecta e classifica atividades (em pé, sentado, deitado) em vídeos pré-gravados ou em tempo real, utilizando redes da família YOLOv11 e OpenCV. O sistema calcula o tempo gasto em cada atividade e gera um relatório ao final do vídeo.

## Metodologia
- Identificar posições de pessoas em um vídeo;
- Contar o número de pessoas em cada posição;
- Calcular o tempo gasto em cada posição;
- Calcular o tempo total gasto com o monitoramento;
- Calcular a taxa de detecção de pessoas em cada posição;

## Resultados

- Contagem de pessoas em cada posição;
- Tempo gasto em cada posição;
- Tempo total gasto com o monitoramento;
- Taxa de detecção de pessoas em cada posição;

## Bibliotecas Utilizadas
- OpenCV
- YOLOv11

## Como instalar ou utilizar o programa

- Baixe o programa:

```
wget hhttps://github.com/ic-2025/atividade_YOLO/releases/download/v0.0.1-latest/v0.0.1-latest.zip
```

- Descompacte o arquivo:

```
unzip v0.0.1-latest.zip
```

- Navegue para a pasta atividade_YOLO-v0.0.1-latest:

```
cd atividade_YOLO-v0.0.1-latest
```

- Crie um ambiente virtual:

    - Para Linux:

    ```
    python3 -m venv venv
    source venv/bin/activate
    ```
    - Para Windows:

    ```
    python -m venv venv
    venv\Scripts\activate
    ```

- Instale as bibliotecas necessárias:

```
pip install -r requirements.txt
```


- Execute o programa:

```
python main.py
```