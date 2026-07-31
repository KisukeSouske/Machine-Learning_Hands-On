# kai

Biblioteca criada para consolidar estudos de regressão linear e Machine Learning,
acompanhada de uma interface gráfica para configurar e acompanhar treinamentos.

## Estrutura

```
kai/
  metrics.py          # métricas de erro e gradiente do MSE
  model.py            # Model: gradiente descendente em mini-lotes
  preprocessing.py    # padronização (Z-score)
  visualization.py    # gráficos matplotlib (recebem o estilo do tema)
  themes/             # sistema de temas
    base.py             contratos: Theme, Palette, FontSet, ChartStyle
    default.py          tema Windows 7 / Aero
    retro_os.py         tema OS retrô / clássico
  gui/                # aplicação
    app.py              janela principal (layout e eventos)
    widgets.py          widgets temáveis
    controller.py       thread de treino + relatório de resultados
    state.py            estado tipado da sessão
    helpers.py          funções puras
tests/                # 107 testes (pytest)
main.py               # ponto de entrada
```

## Instalação

```bash
pip install -r requirements.txt
```

## Interface gráfica

```bash
python main.py
```

O tema é escolhido em `main.py` (constante `THEME`) ou por linha de comando:

```bash
python main.py retro_os
```

Temas disponíveis: `default` (Windows 7 / Aero) e `retro_os` (OS clássico).

## Uso como biblioteca

```python
from kai import loss, mean_absolute_error, r_squared

y_true = [1.0, 2.0, 3.0, 4.0]
y_pred = [1.1, 1.9, 3.2, 3.8]

loss(y_true, y_pred)                 # soma dos erros absolutos (L1)
mean_absolute_error(y_true, y_pred)  # erro absoluto médio (MAE)
r_squared(y_true, y_pred)            # coeficiente de determinação
```

Treinando um modelo:

```python
from kai.model import Model

model = Model("data_valve_calibration.csv", label_column="Vazao_L_min")
model.start_training(
    ["Abertura_Valvula_Percentual"],
    learning_rate=0.05,
    standardize_features=True,   # ver "Escolhendo a taxa de aprendizagem"
    random_state=42,             # opcional: torna a execução reprodutível
    show_plot=False,
)
model.predict([[75.0]])          # entrada em escala ORIGINAL
```

## Escolhendo a taxa de aprendizagem

A taxa de aprendizagem depende da escala das características, e essa dependência é
forte. O gradiente descendente só é estável enquanto `learning_rate < 2/λmax`, onde
`λmax` é o maior autovalor da Hessiana do MSE. Padronizar as características leva o
número de condição da Hessiana para perto de 1, o que eleva esse teto em ordens de
grandeza.

Medido no dataset `data_valve_calibration.csv`:

| Modo | Condição da Hessiana | Teto de estabilidade | Taxa recomendada |
|---|---|---|---|
| Cru (sem padronizar) | ~17.000 | `lr < 0.00035` | `0.0003` |
| Padronizado (Z-score) | ~1 | `lr < 1.0` | `0.05` a `0.5` |

Por isso a interface **redefine a taxa de aprendizagem ao ligar/desligar a
padronização**: uma taxa afinada para colunas cruas fica milhares de vezes pequena
demais depois do Z-score, e o treino consome todas as épocas sem convergir.

## Critério de parada

O treino encerra quando a norma do gradiente cai a uma fração de seu valor inicial:

```
||∇L|| ≤ tolerance × ||∇L_inicial||
```

Por ser **relativo**, o critério independe das unidades de `y` e das características —
multiplicar o alvo por 1000 não altera o número de épocas. O padrão é `1e-4`; valores
menores treinam por mais tempo e chegam mais perto do ótimo.

Um critério baseado apenas na variação da perda entre épocas não é suficiente: em
problemas mal condicionados a perda pode estagnar enquanto os parâmetros ainda estão
longe do ótimo (foi o que a auditoria detectou — ver `docs/AUDITORIA.md`).

## Métricas

Todas as métricas aceitam qualquer *array-like* (listas, tuplas, `pandas.Series`,
`numpy.ndarray`) e são calculadas **in-sample**: como não há divisão treino/teste, elas
medem qualidade de ajuste, não capacidade de generalização.

Casos degenerados levantam `ValueError` em vez de devolver `inf`/`-inf`:
`r_squared` com alvo constante, `adjusted_r_squared` e `f_statistic` com
`n_samples ≤ n_features + 1`, e `f_statistic` com ajuste perfeito.

## Auditoria numérica

As técnicas foram validadas contra scikit-learn, contra a solução fechada por equações
normais e contra as fórmulas de *An Introduction to Statistical Learning* (eq. 3.17,
3.23 e 6.4) e do Cap. 11 (EST-027). O relatório completo está em
[docs/AUDITORIA.md](docs/AUDITORIA.md), e os achados estão fixados como testes de
regressão em `tests/test_audit_fixes.py`.

## Rodando os testes

```bash
python -m pytest
```
