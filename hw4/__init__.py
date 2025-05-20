import json
import random
import requests
import pickle
import os
import webbrowser
from operator import itemgetter
import folium


# === Настройки ===
MAX_TIME_SECONDS = 3600  # Максимальное время (в секундах)
GENERATIONS = 100
POPULATION_SIZE = 50
PROFILE = "walking"  # walking, driving, cycling
USE_CACHED_MATRIX = False  # использовать кэшированную матрицу


# === 1. Загрузка точек ===
def load_points(filename='points.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


# === 2. Получение матрицы времени между точками (OSRM) ===
def get_time_matrix(points, use_cache=False, use_mock=False):
    if use_cache:
        try:
            with open('time_matrix.pkl', 'rb') as f:
                print("Матрица времени загружена из кэша.")
                return pickle.load(f)
        except FileNotFoundError:
            print("Кэшированная матрица не найдена.")

    if use_mock:
        n = len(points)
        matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j] = matrix[j][i] = random.randint(60, 600)  # 1-10 минут
        print("Используется тестовая матрица (mock).")
        return matrix

    base_url = "https://router.project-osrm.org/table/v1"
    coords = ";".join([f"{p['lon']},{p['lat']}" for p in points])
    n = len(points)

    url = f"{base_url}/{PROFILE}/{coords}?sources=all&destinations=all&annotations=duration"

    print("Получение матрицы времени через OSRM...")
    response = requests.get(url)

    try:
        data = response.json()
        if "durations" not in data:
            raise ValueError(f"Ошибка от OSRM: {data}")

        durations = data["durations"]

        if len(durations) != n or len(durations[0]) != n:
            raise ValueError("Неверный формат ответа от OSRM")

        print("Матрица времени успешно получена.")

        # Сохранить в кэш
        with open('time_matrix.pkl', 'wb') as f:
            pickle.dump(durations, f)

        return durations

    except Exception as e:
        print("Ошибка при получении матрицы времени:", response.text)
        raise e


# === 3. Оценка маршрута ===
def evaluate_route(route, time_matrix, points, max_time):
    total_time = 0
    total_priority = 0
    for i in range(len(route) - 1):
        total_time += time_matrix[route[i]][route[i + 1]]
        total_priority += points[route[i]]['priority']
    total_priority += points[route[-1]]['priority']  # Добавляем последнюю точку

    if total_time > max_time:
        return -1  # Не укладывается во временные рамки
    return total_priority


# === 4. Генетический алгоритм ===
def genetic_algorithm(points, time_matrix, max_time):
    n = len(points)
    population = [random.sample(range(n), n) for _ in range(POPULATION_SIZE)]

    for gen in range(GENERATIONS):
        scored_pop = [(ind, evaluate_route(ind, time_matrix, points, max_time)) for ind in population]
        scored_pop = [(x[0], x[1]) for x in scored_pop if x[1] > 0]

        if not scored_pop:
            print("Нет допустимых маршрутов. Генерируем новые.")
            scored_pop = [(random.sample(range(n), n), 1) for _ in range(int(POPULATION_SIZE * 0.2))]

        scored_pop.sort(key=itemgetter(1), reverse=True)
        selected = [x[0] for x in scored_pop[:int(POPULATION_SIZE * 0.2)]]
        next_gen = selected.copy()

        while len(next_gen) < POPULATION_SIZE:
            if len(selected) >= 2:
                p1, p2 = random.sample(selected, 2)
            else:
                p1 = random.choice(population)
                p2 = random.choice(population)

            cut = random.randint(1, n - 2)
            child = p1[:cut] + [gene for gene in p2 if gene not in p1[:cut]]
            if len(child) < n:
                missing = [gene for gene in range(n) if gene not in child]
                child += missing
            next_gen.append(child[:n])

        # Мутация
        for i in range(len(next_gen)):
            if random.random() < 0.1:
                a, b = random.sample(range(n), 2)
                next_gen[i][a], next_gen[i][b] = next_gen[i][b], next_gen[i][a]

        population = next_gen

    best_route = max(population, key=lambda r: evaluate_route(r, time_matrix, points, max_time))
    return best_route


# === 5. Функция отрисовки маршрута на интерактивной карте с Folium ===
def plot_route_on_map(points, route):
    if not route:
        print("Маршрут пуст.")
        return None

    first_point = points[route[0]]
    map_center = [first_point["lat"], first_point["lon"]]
    m = folium.Map(location=map_center, zoom_start=13)

    for idx in route:
        point = points[idx]
        lat = point["lat"]
        lon = point["lon"]
        name = point["name"]
        priority = point["priority"]

        # Цвет маркера по приоритету
        if priority >= 8:
            color = "red"
        elif priority >= 5:
            color = "orange"
        else:
            color = "green"

        folium.Marker(
            [lat, lon],
            popup=f'{name}<br>Приоритет: {priority}',
            icon=folium.Icon(color=color, icon="info-sign")
        ).add_to(m)

    # Рисуем маршрут
    route_coords = [[points[idx]["lat"], points[idx]["lon"]] for idx in route]
    folium.PolyLine(route_coords, color="blue", weight=2.5, opacity=0.7).add_to(m)

    # Сохраняем в HTML
    map_file = 'route_map.html'
    m.save(map_file)
    print(f"Карта сохранена как {map_file}")
    return m


# === 6. Генерация ссылки на статическую карту OpenStreetMap через staticmap.osm.tools ===
def build_osm_staticmap_url(best_route, points):
    base_url = "https://staticmap.osm.tools/api/v1/getmap?"

    size = "800x600"
    zoom = "15"

    markers = []
    path_coords = []

    for idx in best_route:
        lat = points[idx]["lat"]
        lon = points[idx]["lon"]
        priority = points[idx]["priority"]

        # Цвет маркера по приоритету
        if priority >= 8:
            color = "red"
        elif priority >= 5:
            color = "orange"
        else:
            color = "green"

        markers.append(f"marker=color:{color}|{lon},{lat}")
        path_coords.append(f"{lon},{lat}")

    marker_str = "&".join(markers)
    path_str = f"path=color:blue|weight:5|{','.join(path_coords)}"

    full_url = f"{base_url}size={size}&zoom={zoom}&{marker_str}&{path_str}"
    return full_url


# === 7. Основная функция запуска ===
def main():
    print("Загрузка точек...")
    points = load_points()
    print(f"Найдено {len(points)} точек.")

    print("Получение матрицы времени...")
    time_matrix = get_time_matrix(points, use_cache=USE_CACHED_MATRIX, use_mock=False)

    print("Запуск генетического алгоритма...")
    best_route = genetic_algorithm(points, time_matrix, MAX_TIME_SECONDS)

    total_time = sum(time_matrix[best_route[i]][best_route[i+1]] for i in range(len(best_route)-1))
    total_priority = sum(points[idx]['priority'] for idx in best_route)

    print("\nЛучший найденный маршрут:")
    for idx in best_route:
        print(f"- {points[idx]['name']} (приоритет: {points[idx]['priority']})")
    print(f"\nОбщее время: {total_time} секунд (~{round(total_time / 60, 1)} мин)")
    print(f"Общий приоритет: {total_priority}")

    print("\nГенерирую ссылку на карту OpenStreetMap...")
    osm_map_url = build_osm_staticmap_url(best_route, points)
    print("Открыть маршрут на карте:")
    print(osm_map_url)
    webbrowser.open(osm_map_url)

    print("\nСоздаём интерактивную карту с маршрутом...")
    m = plot_route_on_map(points, best_route)
    if m:
        webbrowser.open('route_map.html')


if __name__ == "__main__":
    main()