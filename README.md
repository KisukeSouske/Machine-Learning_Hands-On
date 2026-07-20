# kai

Biblioteca criada para consolidar estudos de regressão linear e Machine Learning.

## Estrutura

```
kai/
  __init__.py     # API pública do pacote
  metrics.py      # métricas de erro (loss, MAE, ...)
tests/
  test_metrics.py # testes unitários (pytest)
```

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

```python
from kai import calculate_loss, mean_absolute_error

y_true = [1.0, 2.0, 3.0]
y_pred = [1.1, 1.9, 3.2]

calculate_loss(y_true, y_pred)       # soma dos erros absolutos
mean_absolute_error(y_true, y_pred)  # erro absoluto médio (MAE)
```

## Rodando os testes

```bash
python -m pytest
```
