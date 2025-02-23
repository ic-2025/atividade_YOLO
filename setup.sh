#bash /usr/bin/bash

mkder ./equip && cd ./equip



unzip beta.zip

rm beta.zip

cd atividade_YOLO-beta

mkder models

wget -0 models/best.pt https://github.com/ic-2025/atividade_YOLO/releases/download/beta/best.pt

pip install -r requirements.txt
