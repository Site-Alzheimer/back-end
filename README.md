# API de Classificacao de Alzheimer por MRI

API FastAPI para classificar exames de ressonancia magnetica cerebral em formato NIfTI (`.nii` ou `.nii.gz`) usando dois modelos CNN:

- `modelo_cnn1_08_maio.h5`: localizacao da regiao de interesse
- `CNN-4-2023-1.h5`: classificacao Alzheimer/Normal

## Requisitos

Para rodar pelo Docker:

- Docker instalado
- Porta `8000` livre
- Pasta `models/` montada como volume; ela pode estar vazia no primeiro startup

Para rodar sem Docker:

- Python `3.10`
- Dependencias do `requirements.txt`

> Recomendado: use Docker. O TensorFlow deste projeto usa versoes antigas e pode nao instalar corretamente em Python mais novo, como Python 3.12.

## Modelos CNN

Os modelos nao devem ir para dentro da imagem Docker. Isso deixa o build mais leve e evita versionar arquivos grandes.

O projeto ja consegue baixar os modelos automaticamente do Google Drive no startup, caso eles ainda nao existam no disco. Se o arquivo ja existir, o download e ignorado.

Por padrao, a API procura os modelos nestes caminhos:

```text
models/modelo_cnn1_08_maio.h5
models/CNN-4-2023-1.h5
```

Voce pode alterar esses caminhos com variaveis de ambiente:

```text
MODEL_ROI_PATH
MODEL_CLF_PATH
```

E tambem pode trocar as URLs de download:

```text
MODEL_ROI_URL
MODEL_CLF_URL
```

As URLs ficam em um arquivo `.env` local, que nao deve ser commitado. Use o arquivo `.env.example` como modelo:

```bash
cp .env.example .env
```

Depois edite o `.env` e preencha:

```text
MODEL_ROI_PATH=/models/modelo_cnn1_08_maio.h5
MODEL_CLF_PATH=/models/CNN-4-2023-1.h5
MODEL_ROI_URL=cole_a_url_do_modelo_roi_aqui
MODEL_CLF_URL=cole_a_url_do_modelo_classificador_aqui
```

O `.gitignore` ja ignora `.env`, entao essas URLs nao vao para o repositorio.

### Baixar Manualmente

Se quiser baixar antes de subir a API, crie a pasta `models/`:

```bash
mkdir -p models
```

Baixe os dois arquivos pelo Google Drive e coloque dentro de `models/`:

```text
models/
├── modelo_cnn1_08_maio.h5
└── CNN-4-2023-1.h5
```

Opcionalmente, voce pode baixar pelo terminal usando `gdown`:

```bash
python3 -m pip install --user gdown

mkdir -p models

python3 -m gdown "cole_a_url_do_modelo_classificador_aqui" \
  -O models/CNN-4-2023-1.h5

python3 -m gdown "cole_a_url_do_modelo_roi_aqui" \
  -O models/modelo_cnn1_08_maio.h5
```

O `.dockerignore` deve manter `*.h5`, para que os modelos continuem fora da imagem.

## Estrutura Esperada

Na raiz do projeto, os arquivos principais devem estar assim:

```text
back-end/
├── main.py
├── engine.py
├── model_loader.py
├── requirements.txt
├── Dockerfile
└── models/
    ├── modelo_cnn1_08_maio.h5
    └── CNN-4-2023-1.h5
```

## Instalacao com Docker

Entre na pasta do projeto:

```bash
cd "/home/diego-ribeiro/Área de trabalho/Programação/IA - Alzheimer/back-end"
```

Construa a imagem:

```bash
docker build -t alzheimer-api .
```

Rode a API montando a pasta `models/` como volume.

Se a pasta `models/` ja tem os dois `.h5`, use somente leitura:

```bash
docker run --rm -d \
  --name alzheimer-api \
  -p 8000:8000 \
  -v "$PWD/models:/models:ro" \
  --env-file .env \
  alzheimer-api
```

Se a pasta `models/` ainda esta vazia e voce quer que o container baixe os modelos automaticamente do Google Drive, use o volume com escrita:

```bash
mkdir -p models

docker run --rm -d \
  --name alzheimer-api \
  -p 8000:8000 \
  -v "$PWD/models:/models" \
  --env-file .env \
  alzheimer-api
```

Na primeira vez, o startup pode demorar mais porque os dois modelos serao baixados. Nas proximas execucoes, a API detecta os arquivos em `models/` e nao baixa novamente.

A API ficara disponivel em:

```text
http://localhost:8000
```

Verifique se subiu corretamente:

```bash
curl http://localhost:8000/healthz
```

Resposta esperada:

```json
{
  "status": "ok",
  "models_loaded": true
}
```

Para acompanhar os logs:

```bash
docker logs -f alzheimer-api
```

Para parar a API:

```bash
docker stop alzheimer-api
```

## Rodando Localmente sem Rebuild

Sempre que alterar apenas os modelos ou trocar os arquivos `.h5`, nao precisa reconstruir a imagem. Basta parar e subir o container novamente, porque os modelos sao lidos da pasta local `models/`.

Se quiser forcar novo download, apague os arquivos de `models/` e suba o container novamente com o volume em modo de escrita.

## Instalacao sem Docker

Use Python `3.10`.

Entre na pasta do projeto:

```bash
cd "/home/diego-ribeiro/Área de trabalho/Programação/IA - Alzheimer/back-end"
```

Crie um ambiente virtual:

```bash
python3.10 -m venv .venv
```

Ative o ambiente:

```bash
source .venv/bin/activate
```

Instale as dependencias:

```bash
pip install -r requirements.txt
```

Crie o arquivo `.env`:

```bash
cp .env.example .env
```

No modo sem Docker, edite o `.env` para usar caminhos locais, sem a barra inicial `/`, porque `/models` e o caminho interno do container:

```text
MODEL_ROI_PATH=models/modelo_cnn1_08_maio.h5
MODEL_CLF_PATH=models/CNN-4-2023-1.h5
MODEL_ROI_URL=cole_a_url_do_modelo_roi_aqui
MODEL_CLF_URL=cole_a_url_do_modelo_classificador_aqui
```

Rode a API. Se os modelos ainda nao existirem, eles serao baixados automaticamente. Carregue as variaveis do `.env` antes:

```bash
set -a
source .env
set +a

uvicorn main:app --host 0.0.0.0 --port 8000
```

Durante o startup, a API carrega os dois modelos `.h5`. Se algum modelo estiver faltando, o servidor nao inicia corretamente.

## Variaveis de Ambiente no VS Code

Para usar no terminal integrado do VS Code:

1. Crie um arquivo chamado `.env` na raiz do projeto.
2. Copie o conteudo de `.env.example`.
3. Preencha `MODEL_ROI_URL` e `MODEL_CLF_URL` com as URLs do Google Drive.
4. Rode os comandos Docker com `--env-file .env`.

Para rodar sem Docker no terminal integrado:

```bash
set -a
source .env
set +a
uvicorn main:app --host 0.0.0.0 --port 8000
```

No modo sem Docker, edite o `.env` para usar caminhos locais como `models/...` em vez de `/models/...`, porque `/models` e o caminho interno do container.

Se voce usa o modo `Run and Debug` do VS Code, pode criar `.vscode/launch.json` e apontar para o `.env`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "FastAPI",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": ["main:app", "--host", "0.0.0.0", "--port", "8000"],
      "envFile": "${workspaceFolder}/.env"
    }
  ]
}
```

## Producao no Render

No Render, a ideia deve ser a mesma: a imagem Docker fica sem os modelos, e os `.h5` ficam em armazenamento persistente.

Fluxo recomendado:

1. Suba o projeto para um repositorio Git.
2. Crie um `Web Service` no Render usando Docker.
3. Configure um Persistent Disk montado em:

```text
/models
```

4. Coloque os dois arquivos `.h5` nesse disco persistente.
5. Configure as variaveis de ambiente:

```text
MODEL_ROI_PATH=/models/modelo_cnn1_08_maio.h5
MODEL_CLF_PATH=/models/CNN-4-2023-1.h5
MODEL_ROI_URL=URL_DO_GOOGLE_DRIVE_DO_MODELO_ROI
MODEL_CLF_URL=URL_DO_GOOGLE_DRIVE_DO_MODELO_CLASSIFICADOR
```

6. No primeiro startup, se os modelos ainda nao estiverem no disco, a API baixa automaticamente do Google Drive para `/models`.
7. O Render define a variavel `PORT` automaticamente. O `Dockerfile` ja usa essa porta:

```dockerfile
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
```

### Como colocar os modelos no Render

Voce tem duas opcoes.

Opcao 1: deixar a API baixar automaticamente.

1. Crie o Persistent Disk em `/models`.
2. Configure `MODEL_ROI_PATH` e `MODEL_CLF_PATH`.
3. Garanta que os links do Google Drive estejam publicos para quem tem o link.
4. Faca deploy.
5. A primeira inicializacao baixa os modelos; as proximas reutilizam os arquivos do disco.

Opcao 2: baixar manualmente pelo Shell do Render.

1. Criar o servico com Persistent Disk em `/models`.
2. Abrir o Shell do servico no painel do Render.
3. Baixar os modelos para `/models` usando `gdown`, `scp` ou outro metodo de transferencia.

Exemplo pelo Shell do Render:

```bash
cd /models

pip install gdown

gdown "URL_DO_GOOGLE_DRIVE_DO_MODELO_CLASSIFICADOR" -O CNN-4-2023-1.h5
gdown "URL_DO_GOOGLE_DRIVE_DO_MODELO_ROI" -O modelo_cnn1_08_maio.h5
```

Depois reinicie o servico. A API deve iniciar carregando os modelos a partir de `/models`.

Observacoes importantes sobre Render:

- Sem Persistent Disk, o filesystem do servico e efemero; arquivos baixados somem em restart/redeploy.
- Persistent Disk fica disponivel em runtime, nao durante o build.
- Um servico com Persistent Disk nao escala para multiplas instancias usando o mesmo disco.
- A imagem Docker continua leve porque os `.h5` nao entram no build.

## Endpoints

### Health Check

```http
GET /healthz
```

Exemplo:

```bash
curl http://localhost:8000/healthz
```

### Classificacao de MRI

```http
POST /v1/predict/alzheimer
```

O endpoint espera um arquivo no campo `file`, em formato `.nii` ou `.nii.gz`.

Exemplo com `curl`:

```bash
curl -F "file=@/home/diego-ribeiro/Downloads/AD_002_S_0619.nii" \
  http://localhost:8000/v1/predict/alzheimer
```

Outro exemplo:

```bash
curl -F "file=@/home/diego-ribeiro/Downloads/CN_002_S_0295.nii" \
  http://localhost:8000/v1/predict/alzheimer
```

## Testando pelo Postman

### Testar `/healthz`

1. Abra o Postman.
2. Crie uma nova requisicao.
3. Selecione o metodo `GET`.
4. Use a URL:

```text
http://localhost:8000/healthz
```

5. Clique em `Send`.

### Testar `/v1/predict/alzheimer`

1. Crie uma nova requisicao.
2. Selecione o metodo `POST`.
3. Use a URL:

```text
http://localhost:8000/v1/predict/alzheimer
```

4. Abra a aba `Body`.
5. Selecione `form-data`.
6. Crie um campo chamado exatamente:

```text
file
```

7. Troque o tipo do campo de `Text` para `File`.
8. Selecione um arquivo `.nii` ou `.nii.gz`.
9. Clique em `Send`.

## Exemplo de Resposta

```json
{
  "status": "success",
  "diagnostico": "NORMAL",
  "probabilidade_media": 0.006876706366280192,
  "estatisticas_predicao": {
    "desvio_padrao": 0.0016299878938484781,
    "coeficiente_variacao": 0.0016412744398248906,
    "classificacao_cv": "High precision",
    "predicoes_19_slices": [
      0.0051,
      0.0062
    ]
  },
  "metadados_nifti": {
    "dimensoes": [256, 256, 166],
    "espacamento": [0.9475275278091431, 0.9442049860954285, 1.202307105064392]
  },
  "metadados_varredura_cnn1": {
    "roi_scores": [],
    "limite_inferior": 141,
    "limite_superior": 144,
    "slice_central": 142
  },
  "metadados_extracao_cnn2": {
    "inicio_cranio_y": 9
  }
}
```

## Problemas Comuns

### Erro ao carregar modelos

Verifique se estes arquivos existem na pasta `models/` localmente ou em `/models` no Render:

```text
models/modelo_cnn1_08_maio.h5
models/CNN-4-2023-1.h5
```

Verifique tambem se as variaveis apontam para os caminhos corretos:

```text
MODEL_ROI_PATH
MODEL_CLF_PATH
```

### Porta 8000 ocupada

Use outra porta local:

```bash
docker run --rm -d \
  --name alzheimer-api \
  -p 8001:8000 \
  -v "$PWD/models:/models:ro" \
  --env-file .env \
  alzheimer-api
```

Nesse caso, acesse:

```text
http://localhost:8001
```

### Python local nao instala TensorFlow

Use Docker. O projeto foi validado rodando em container com Python `3.10`.

### Postman retorna erro 422

Confira se o campo do `form-data` se chama exatamente `file` e se o tipo esta como `File`.

### Arquivo rejeitado

Somente arquivos `.nii` e `.nii.gz` sao aceitos.
