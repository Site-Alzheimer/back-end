# Exames de exemplo

Esta pasta é lida pela API (`AMOSTRAS_DIR`, padrão `amostras/`) e alimenta o carrossel do
front. Só este arquivo é versionado: os exames e as miniaturas são dados de pacientes e ficam
fora do git (veja o `.gitignore`).

Antes de publicar qualquer exame, confira a licença. O termo de uso do ADNI proíbe
redistribuir os dados; para o site público use bases que permitam, como a IXI (CC BY-SA 3.0).

## `amostras.json`

```json
{
  "amostras": [
    {
      "id": "adni-002-s-0619",
      "rotulo": "Exame de exemplo 1",
      "fonte": "ADNI",
      "diagnostico_clinico": "AD",
      "licenca": "ADNI Data Use Agreement — uso restrito, não redistribuir",
      "arquivo": "AD_002_S_0619.nii.gz"
    }
  ]
}
```

- `arquivo`: caminho relativo a esta pasta (`.nii` ou `.nii.gz`).
- `diagnostico_clinico`: rótulo da base (`AD`, `CN`, `MCI`), só informativo.

Depois de editar o manifesto, gere as miniaturas:

```bash
python scripts/gerar_miniaturas.py amostras
```
