# Zmeika

**Краткое описание проекта.**
Это модифицированная версия классической змейки на языке python, используя pygame и numpy. Эта версия является не конечной, а только разрабатывается.

## Что добавил?

- Простую кастомизацию цвета
- Возможность усложнения/упрощения игры
- Подсчёт очков зависящих от сложности

## Кратко о игровом цикле

Вы появляетсь на поле с едой, передвигаясь через еду, длина змейки увеличивается, а вы получаете очки. Как только вся еда на поле заканчивается она появляется заново, а вы получаете бонусные очки, но ваша змейка не теряет своей длины. Игра продолжается, пока змейка куда-нибудь не врежется.

**Примечания к игре.**
Так как изначально открывается не игра, а консоль, после запуска игры, нужно кликнуть мышкой, чтобы окно стало активно. "Settings" будет неактивно (тогда бы не было необходимости в консоле). После "Start", появится поле, а игра будет на паузе, чтобы начать игру нужно нажать space. Также можно поставить игру на паузу нажав тот же space.

## Как запустить?
```bash
git clone https://github.com/saigedoc/zmeika
pip install -r requirements.txt
python start.py
```
Тесты были проведены только на версии Python 3.10 на Windows 10.

## Настройки

Основной интерфейс консольный и все настройки меняются через неё. Пример настроек с объяснением:
```zmeika settings
FPS: 120                               # Ограничение FPS (False, чтобы убрать ограничение)
background: (0, 0, 0)                  # RGB цвет для заднего фона
color: [255, 0, 255]                   # RGB цвет для основных элементов
size: [50, 50]                         # Размер игрового поля (единица - одна голова змейки)
indent: 1                              # Отступ от края окна (единица - одна голова змейки)
display_size: [1920, 1080]             # Разрешение окна
start_position: False                  # Позиция в начале игры [x,y], False - будет выбрана середина поля
speed: 0.02                            # Скорость змейки (0.02 стандартная скорость для поля 50/50
start_food: False                      # Можно указать позицию стартовой еды [x, y, SIZE] SIZE - размер еды (единица - одна голова змейки)
food_gen: 200.0                        # Количество появляющейся еды
food_size_chance: [70, 15, 10, 3, 2]   # Шанс появления еды разных размеров, в данном случае: 70% - size 1, 15, size 2, и тд.
food_color: (255, 255, 0)              # RGB цвет еды
block_type: 1                          # Нерабочая опция в данный момент, меняет тип блока змейки
fullscreen: True                       # Открывает игру в полнноэкранном режиме
label: 10                              # нерабочая опция в данный момент, отвечает за ИИ
n_bl: 1.0                              # нерабочая опция в данный момент, отвечает за ИИ
load_config: True                      # Загружает сохранённые настройки при запуске, вместо значений по умолчанию.
```

## Формула подсчёта очков

Если не углубляться в вычисления, то чем сложнее выбраны настройки, тем больше очков:
- Больше уровней прошёл (количество подряд целиком съеденой всей еды на поле) = сложнее игра = больше очков
- Больше скорость = сложнее игра = больше очков
- Длиннее змейка = сложнее игра = больше очков
- Меньше поле = сложнее игра = больше очков
- Больше еды = сложнее игра = больше очков
- Меньше еды Больших размеров = сложнее игра = больше очков
