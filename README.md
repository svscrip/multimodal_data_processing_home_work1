# ДЗ1. Мультимодальная обработка данных

Домашнее задание №1: механизм внимания изнутри и поиск семантических ошибок в реализации
декодер-трансформера (MiniGPT). Оба ноутбука выполнены, все проверки зелёные, выводы ячеек
сохранены в файлах.

[![Open dz1-attention.ipynb in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/svscrip/multimodal_data_processing_home_work1/blob/main/dz1-attention.ipynb)
[![Open dz1-bughunt.ipynb in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/svscrip/multimodal_data_processing_home_work1/blob/main/dz1-bughunt.ipynb)

## Что в репозитории

| Файл | Что это |
|---|---|
| `dz1-attention.ipynb` | Шаг 1. Внимание своими руками: дописаны четыре функции, 17 из 17 проверок |
| `dz1-bughunt.ipynb` | Шаг 2. Охота на баги (вариант seed 42): найдены и исправлены три семантические ошибки, 17 из 17 проверок, кривая потерь и пример генерации |
| `dz1-attention.backup.ipynb` | Исходное состояние шага 1 до правок (с `NotImplementedError`) — для сравнения «до/после» |
| `dz1-bughunt.backup.ipynb` | Исходное состояние шага 2 с тремя подсаженными багами — для сравнения «до/после» |

## Шаг 1. Внимание своими руками

Дописаны четыре функции:

- **`split_heads`** — `(B, L, d_model) → (B, n_heads, L, d_head)`. Последняя ось сначала
  разворачивается в `(n_heads, d_head)` и только потом переставляются оси, поэтому голова `h`
  получает свой непрерывный кусок каналов `[h·d_head : (h+1)·d_head]`. Обратный порядок дал бы
  ту же форму, но перемешал бы каналы между головами.
- **`merge_heads`** — точное обращение: `transpose(1, 2).contiguous().view(B, L, -1)`.
  `contiguous()` обязателен, потому что после `transpose` тензор не непрерывен в памяти.
- **`scaled_dot_product_attention`** — `softmax(QKᵀ / √d_k) · V`. Маска из нулей и единиц
  (единица = «смотреть можно») применяется к логитам **до** softmax подстановкой `-inf`;
  dropout применяется к весам внимания уже после softmax; возвращается `(выход, веса)`.
- **`TransformerBlock.forward`** — pre-norm: `x = x + attn(norm1(x))`, затем
  `x = x + ffn(norm2(x))`, возвращается `(x, веса внимания)`.

Какое из четырёх решений в `scaled_dot_product_attention` оказалось самым неочевидным —
разобрано в последней ячейке ноутбука.

## Шаг 2. Охота на баги

Код запускался и обучался, но три ошибки меняли то, что вычисляется:

| # | Где | Механизм поломки | Исправление |
|---|---|---|---|
| 1 | `sinusoidal_positional_encoding` | Нумерация позиций начиналась с 1 вместо 0: все фазы сдвигаются на шаг, позиция 0 перестаёт быть опорной, последняя позиция контекста остаётся без своего кода, относительный сдвиг `i - j` не соответствует разности фаз | `torch.arange(1, length + 1, …)` → `torch.arange(length, …)` |
| 2 | `LayerNorm.forward` | Статистики считались по оси батча, а не по признакам токена — это batch norm без бегущих статистик. При батче из одного примера (генерация) дисперсия по батчу равна нулю, `(x - mean)/√eps` обнуляет представление: логиты нулевые, распределение равномерное, текст вырождается | `dim=0` → `dim=-1` в `mean` и `var` |
| 3 | `scaled_dot_product_attention` | Softmax брался по оси запросов, а не ключей: нормировались столбцы, сумма весов по ключам перестала быть равной 1, маска не давала корректных нулей по будущим позициям, информация текла из будущего | `torch.softmax(scores, dim=-2)` → `dim=-1` |

Подробный разбор каждой ошибки (механизм и почему он меняет результат) — в ячейке
«Ваши ответы» ноутбука `dz1-bughunt.ipynb`.

## Результаты прогона

| Ноутбук | Проверки | Что ещё |
|---|---|---|
| `dz1-attention.ipynb` | **17 из 17** | — |
| `dz1-bughunt.ipynb` | **17 из 17** | потери `3.184 → 0.037` за 600 шагов, график потерь и пример генерации сохранены в ноутбуке |

До исправления багов в `dz1-bughunt.ipynb` падали 10 проверок из 17 — это воспроизводится
прогоном `dz1-bughunt.backup.ipynb`.

## Как запустить

### Colab (быстрый путь, ничего устанавливать не нужно)

Прямые ссылки — Colab сам подтянет ноутбук из этого репозитория (он публичный, авторизация не
нужна):

- [dz1-attention.ipynb в Colab](https://colab.research.google.com/github/svscrip/multimodal_data_processing_home_work1/blob/main/dz1-attention.ipynb)
- [dz1-bughunt.ipynb в Colab](https://colab.research.google.com/github/svscrip/multimodal_data_processing_home_work1/blob/main/dz1-bughunt.ipynb)

Или вручную: `File → Open notebook → вкладка GitHub` → в поиске
`svscrip/multimodal_data_processing_home_work1` → выбрать ноутбук.

Затем `Runtime → Run all` (Ctrl+F9). Достаточно CPU-рантайма: модель крошечная, GPU не нужен,
а `torch` и `matplotlib` в Colab уже стоят, так что ячейка установки ничего не делает. Colab
открывает копию файла, репозиторий при запуске не меняется.

Что должно быть в выводе:

1. первая ячейка печатает версию torch;
2. ячейка «Прогон» — 17 строк `ok …`, затем `пройдено 17 из 17` и `Все проверки зелёные.`;
3. в `dz1-bughunt.ipynb` дальше идут шаги обучения (`шаг 0 …` → `итог: 3.184 -> 0.0xx`),
   график потерь и пример сгенерированного текста;
4. красных traceback быть не должно.

Время: `dz1-attention.ipynb` — секунды, `dz1-bughunt.ipynb` — 1–3 минуты на CPU (600 шагов
обучения).

Проверка «до/после»: откройте в Colab `dz1-bughunt.backup.ipynb` и нажмите `Run all` — там
ожидаемо `пройдено 7 из 17` и 10 упавших проверок.

Одна грабля: ячейка обучения в `dz1-bughunt.ipynb` делает `cfg = GPTConfig(...)` и перетирает
глобальный хелпер `cfg(m, **kw)` из ячейки с проверками. Если после обучения повторно запустить
ячейку «Прогон», получите `TypeError: 'GPTConfig' object is not callable` — лечится
`Runtime → Restart session and run all`.

Чтобы забрать результат с выводами: `File → Download → Download .ipynb` или
`File → Save a copy in Drive`.

### Локально или в VS Code

```bash
python -m pip install torch matplotlib nbformat nbclient ipykernel
```

Открыть ноутбук в Jupyter или VS Code (ядро «Python 3») и выполнить все ячейки. Проверки
открыты и лежат прямо в ноутбуках: ячейка «Прогон» печатает «пройдено N из N» и
«Все проверки зелёные».

Проверено на Python 3.12.10, torch 2.13.0+cpu, matplotlib 3.11.0.
