# Exames de exemplo

Esta pasta é lida pela API (`AMOSTRAS_DIR`, padrão `amostras/`) e alimenta o carrossel do
front. Só este arquivo é versionado: os exames e as miniaturas são dados de pacientes e ficam
fora do git (veja o `.gitignore`).

O arquivo de um exame não sai do servidor: não há rota para baixá-lo, a análise roda na API e o
front mostra só as imagens que ela devolve. Assim os exames do ADNI ficam só no carrossel, para
testar o site, como combinado com o Dr. Saraiva (19/11/2026).

Em hospedagem sem disco próprio (o Space do Hugging Face), a API baixa esta pasta inteira de
`AMOSTRAS_URL` ao ligar: um zip com o `amostras.json`, os exames e as miniaturas na raiz.

## `amostras.json`

```json
{
  "amostras": [
    {
      "id": "adni-005-s-0221",
      "rotulo": "Exame de exemplo 1",
      "fonte": "ADNI",
      "diagnostico_clinico": "AD",
      "idade": 68,
      "sexo": "M",
      "licenca": "ADNI Data Use Agreement — uso restrito, não redistribuir",
      "arquivo": "005_S_0221.nii.gz"
    }
  ]
}
```

- `arquivo`: caminho relativo a esta pasta (`.nii` ou `.nii.gz`).
- `diagnostico_clinico`: rótulo da base (`AD`, `CN`, `MCI`), só informativo.
- `idade` e `sexo` (`F` ou `M`): opcionais, aparecem no card do carrossel. Os de hoje vêm do
  `ADNI_Pacientes.csv` que o Dr. Saraiva mandou.

Depois de editar o manifesto, gere as miniaturas:

```bash
python scripts/gerar_miniaturas.py amostras
```
