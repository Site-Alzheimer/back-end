"""
scripts/testar_fila.py
----------------------
Teste de ponta a ponta da fila de análises (RabbitMQ) numa API já no ar, com a fila ativa.

    python scripts/testar_fila.py --api http://127.0.0.1:8000 --arquivo amostras/005_S_0221.nii.gz \\
        [--referencia http://127.0.0.1:8001]

Confere:
1. /healthz com a fila e pelo menos um worker;
2. três exames de exemplo enviados juntos: 202 na hora, posições no SSE (`queued`), um exame por
   vez e na ordem de chegada, e o resultado em GET /v1/analises/{job_id};
3. o mesmo exame enviado como arquivo, pela fila e pela rota antiga /v1/predict: mesma resposta;
4. erros: extensão inválida (400 na hora), arquivo corrompido (400 no resultado), job_id repetido
   (409) e análise desconhecida (404);
5. com --referencia (a mesma API sem a fila), respostas idênticas, fora os tempos.

Usa só o `requests`, que já vem na imagem da API (dependência do gdown).
"""

import argparse
import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

ESPERA_MAX_S = 600
falhas = 0


def checar(condicao: bool, mensagem: str) -> bool:
    global falhas
    print(("  ok      " if condicao else "  FALHOU  ") + mensagem, flush=True)
    if not condicao:
        falhas += 1
    return condicao


class Escuta(threading.Thread):
    """Acompanha o SSE de um job e guarda os eventos, com o instante de chegada."""

    def __init__(self, api: str, job_id: str):
        super().__init__(daemon=True)
        self.url = f"{api}/v1/progress/{job_id}"
        self.eventos = []
        self.pronto = threading.Event()
        self.fim = threading.Event()

    def run(self) -> None:
        with requests.get(self.url, stream=True, timeout=(5, ESPERA_MAX_S)) as r:
            for linha in r.iter_lines(decode_unicode=True):
                if not linha or not linha.startswith("data: "):
                    continue
                evento = json.loads(linha[len("data: "):])
                evento["_t"] = time.monotonic()
                self.eventos.append(evento)
                if evento.get("stage") == "ready":
                    self.pronto.set()
                if evento.get("stage") == "result":
                    self.fim.set()
                    return

    def instante(self, etapa: str) -> float:
        return next((e["_t"] for e in self.eventos if e.get("stage") == etapa), float("inf"))

    def posicoes(self) -> list:
        return [e["posicao"] for e in self.eventos if e.get("stage") == "queued"]


def escutar(api: str, job_id: str) -> Escuta:
    escuta = Escuta(api, job_id)
    escuta.start()
    if not escuta.pronto.wait(10):
        raise SystemExit(f"O SSE de {job_id} não mandou o 'ready'")
    return escuta


def sem_tempos(resposta: dict) -> dict:
    return {k: v for k, v in resposta.items() if k != "tempos_ms"}


def esperar_resultado(api: str, job_id: str, escuta: Escuta) -> requests.Response:
    escuta.fim.wait(ESPERA_MAX_S)
    return requests.get(f"{api}/v1/analises/{job_id}", timeout=30)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", required=True, help="API com a fila ativa")
    p.add_argument("--arquivo", required=True, help="Um .nii.gz que também esteja entre as amostras")
    p.add_argument("--referencia", help="A mesma API sem a fila, para comparar as respostas")
    a = p.parse_args()
    api = a.api.rstrip("/")
    params = {"visualizacao": "true", "explicacao": "true"}

    print("1. Saúde")
    saude = requests.get(f"{api}/healthz", timeout=10).json()
    checar("fila" in saude, f"/healthz traz a fila: {saude.get('fila')}")
    checar(saude.get("models_loaded") is True, "há worker conectado (models_loaded)")

    print("2. Três exames de exemplo ao mesmo tempo")
    amostras = requests.get(f"{api}/v1/amostras", timeout=10).json()
    ids = [x["id"] for x in amostras[:3]]
    jobs = [str(uuid.uuid4()) for _ in ids]
    escutas = [escutar(api, j) for j in jobs]
    aceitos = []
    for amostra, job in zip(ids, jobs):
        r = requests.post(f"{api}/v1/analises/amostra/{amostra}", params={**params, "job_id": job}, timeout=30)
        aceitos.append(r)
    checar(all(r.status_code == 202 for r in aceitos), f"POST responde 202: {[r.status_code for r in aceitos]}")
    checar(all(r.headers.get("Location") == f"/v1/analises/{j}" for r, j in zip(aceitos, jobs)),
           "Location aponta para o resultado")
    corpos = [r.json() for r in aceitos]
    checar([c.get("posicao") for c in corpos[1:]] == [2, 3],
           f"posições na resposta: {[c.get('posicao') for c in corpos]}")
    r3 = requests.get(f"{api}/v1/analises/{jobs[2]}", timeout=10)
    checar(r3.status_code == 202 and r3.json().get("estado") == "na_fila",
           f"GET do 3º enquanto espera: {r3.status_code} {r3.json().get('estado')}")

    respostas = [esperar_resultado(api, j, e) for j, e in zip(jobs, escutas)]
    checar(all(r.status_code == 200 for r in respostas), f"GET do resultado: {[r.status_code for r in respostas]}")
    checar(escutas[2].posicoes()[:1] == [3] and escutas[2].posicoes() == sorted(escutas[2].posicoes(), reverse=True),
           f"SSE do 3º: posições {escutas[2].posicoes()}")
    em_ordem = all(escutas[i + 1].instante("start") >= escutas[i].instante("result") for i in range(2))
    checar(em_ordem, "um exame por vez, na ordem de chegada (início do próximo depois do fim do anterior)")
    etapas = [e.get("stage") for e in escutas[0].eventos]
    checar(all(x in etapas for x in ("start", "load", "cnn1", "extract", "cnn2", "done", "result")),
           "SSE com início, as etapas das CNNs e o fim")
    for r in respostas:
        if r.status_code == 200:
            corpo = r.json()
            print(f"          {corpo['diagnostico']:9} {corpo['probabilidade_media'] * 100:6.2f}%")

    print("3. Arquivo enviado: fila, rota antiga e amostra dão o mesmo")
    nome = os.path.basename(a.arquivo)
    amostra = next((x["id"] for x in amostras if x["arquivo"] == nome), None)
    with open(a.arquivo, "rb") as f:
        dados = f.read()
    job = str(uuid.uuid4())
    escuta = escutar(api, job)
    r = requests.post(f"{api}/v1/analises", params={**params, "job_id": job},
                      files={"file": (nome, dados)}, timeout=60)
    checar(r.status_code == 202, f"POST /v1/analises com arquivo: {r.status_code}")
    pela_fila = esperar_resultado(api, job, escuta)
    checar(pela_fila.status_code == 200, f"resultado do arquivo: {pela_fila.status_code}")
    antiga = requests.post(f"{api}/v1/predict/alzheimer", params=params,
                           files={"file": (nome, dados)}, timeout=ESPERA_MAX_S)
    checar(antiga.status_code == 200, f"rota antiga /v1/predict pela fila: {antiga.status_code}")
    if pela_fila.ok and antiga.ok:
        checar(sem_tempos(pela_fila.json()) == sem_tempos(antiga.json()), "fila == rota antiga")
    if amostra and pela_fila.ok:
        da_amostra = next((x for i, x in zip(ids, respostas) if i == amostra), None)
        if da_amostra is not None and da_amostra.ok:
            checar(sem_tempos(pela_fila.json()) == sem_tempos(da_amostra.json()), "arquivo == amostra")

    print("4. Erros")
    r = requests.post(f"{api}/v1/analises", files={"file": ("exame.txt", b"x")}, timeout=30)
    checar(r.status_code == 400, f"extensão inválida: {r.status_code}")
    job = str(uuid.uuid4())
    escuta = escutar(api, job)
    r = requests.post(f"{api}/v1/analises", params={"job_id": job},
                      files={"file": ("corrompido.nii", b"isto nao e um NIfTI")}, timeout=30)
    resultado = esperar_resultado(api, job, escuta)
    checar(r.status_code == 202 and resultado.status_code == 400,
           f"arquivo corrompido: {r.status_code} e depois {resultado.status_code} ({resultado.json().get('detail')})")
    r = requests.post(f"{api}/v1/analises/amostra/{ids[0]}", params={"job_id": jobs[0]}, timeout=30)
    checar(r.status_code == 409, f"job_id repetido: {r.status_code}")
    r = requests.get(f"{api}/v1/analises/{uuid.uuid4()}", timeout=10)
    checar(r.status_code == 404, f"análise desconhecida: {r.status_code}")
    r = requests.get(f"{api}/v1/analises/x", timeout=10)
    checar(r.status_code == 400, f"job_id inválido: {r.status_code}")

    if a.referencia:
        print("5. Sem a fila, as mesmas respostas")
        ref = a.referencia.rstrip("/")
        with ThreadPoolExecutor(3) as ex:
            diretas = list(ex.map(lambda i: requests.post(
                f"{ref}/v1/predict/alzheimer/amostra/{i}", params=params, timeout=ESPERA_MAX_S), ids))
        for i, direta, pela in zip(ids, diretas, respostas):
            checar(direta.ok and pela.ok and sem_tempos(direta.json()) == sem_tempos(pela.json()),
                   f"{i}: direto == fila")

    print()
    print("Tudo certo." if falhas == 0 else f"{falhas} verificação(ões) falharam.")
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()
