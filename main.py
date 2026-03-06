
# %%

import numpy as np
import matplotlib.pyplot as plt

from time import time
from tqdm import tqdm

from numba import njit, prange

np.random.seed(21)

# %%

 ######## SAVE ######## 

save = False
save_filename = 'result.xlsx'

 ######## SAVE ######## 

zerna = np.load('zerna.npy')
save_mean, save_std = 0.648232, 0.28667818805994527 # TEMP
zerna = zerna * save_std + save_mean

# %%

 ######## PARAMS ######## 

# plane
plane_type = 'гладкая' # гладкая, шероховатая, exel
exel_name = 'surf.xlsx'

resolution_width = 300
resolution_height = 300
plane_width = 100 # мкм (X_заг)
plane_height = 100 # мкм (Y_заг)

display_min_z = -30 # мкм, график
display_max_z = 30 # мкм, график

# зерна, (в папке "zerna"!)
zerno_height_min = 5 # мкм, высота зерна, не над колесом, а его вертикальный размер
zerno_height_max = 15 # мкм

zerno_width_min = 5 # мкм, ширина зерна
zerno_width_max = 15 # мкм

zerno_count = 100000 # uint

sechen_step = 5 # мкм, как сильно затупляются зерна
rezok_before = 0 # сколько диском уже пользовались до

# диск
wheel_speed_x = 0 # м/мин (S_пр)
wheel_speed_y = 0 # м/мин (S_п)
wheel_radius = 1000 # мкм (D_кр)
wheel_width = 50 # мкм
zerno_depth = 10 # мкм (t0)

EPS = 1e-5

 ######## PARAMS ######## 

start_pos_x = plane_width / 2 # мкм
start_pos_y = plane_height / 2 # мкм

wheel_speed_x *= 1e6 / 60 # м/мин -> мкм/сек
wheel_speed_y *= 1e6 / 60 # м/мин -> мкм/сек

wheel_height = wheel_radius - zerno_depth
wheel_radius = wheel_radius - zerno_height_max

zerno_width_max = min(zerno_width_max, wheel_width)
zerno_width_min = min(zerno_width_min, wheel_width)
_wheel_width2 = wheel_width / 2

zerno_height_mean = (zerno_height_max + zerno_height_min) / 2
zerno_height_std = (zerno_height_max - zerno_height_min) / 3
zerno_height = lambda: np.clip(zerno_height_mean + np.random.randn(1) * zerno_height_std, zerno_height_min, zerno_height_max).item()

zerno_width_mean = (zerno_width_max + zerno_width_min) / 2
zerno_width_std = (zerno_width_max - zerno_width_min) / 3
zerno_width = lambda: np.clip(zerno_width_mean + np.random.randn(1) * zerno_width_std, zerno_width_min, zerno_width_max).item()

# %%

if wheel_speed_x > wheel_radius:
    raise Exception('ЭТОТ КОД НЕ ПОДДЕРЖИВАЕТ СЛУЧАЙ wheel_speed_x > wheel_radius')

# %%

 ######## PLANE ######## 

if plane_type == 'гладкая':
    plane = np.zeros((resolution_height, resolution_width), dtype = np.float64)
elif plane_type == 'шероховатая':
    plane = np.random.randn(resolution_height, resolution_width) * max(plane_height, plane_width) * 0.01
elif plane_type == 'exel':
    from pandas import read_excel
    exel = read_excel(exel_name, header = None).to_numpy()
else:
    raise KeyError('Нету плоскости типа "%s"' % plane_type)

 ######## PLANE ######## 

X, Y = np.meshgrid(np.linspace(0, 1, resolution_width), np.linspace(0, 1, resolution_height))
X *= plane_width
Y *= plane_height

X = X.ravel()
Y = Y.ravel()

X -= start_pos_x
Y -= start_pos_y
plane -= wheel_height

# %%

@njit(parallel = True, fastmath = True)
def SOLVE_VECTORIZED(a_arr, b_arr, v, eps = EPS):
    n = a_arr.size
    roots = np.empty(n, dtype = np.float64)

    pi_2 = np.pi / 2.0
    pi3_2 = 3.0 * np.pi / 2.0

    for i in prange(n):
        a = a_arr[i]
        b = b_arr[i]

        t = np.pi

        for _ in range(15):
            sin_t = np.sin(t)
            cos_t = np.cos(t)

            f_val = b * sin_t + (v * t - a) * cos_t

            if abs(f_val) < eps:
                break

            f_deriv = (b + v) * cos_t - (v * t - a) * sin_t

            step = f_val / f_deriv
            t_new = t - step

            if t_new <= pi_2 or t_new >= pi3_2:
                if step > 0:
                    t_new = (t + pi_2) / 2.0
                else:
                    t_new = (t + pi3_2) / 2.0
            t = t_new

        if abs(f_val) < eps:
            roots[i] = t
        else:
            roots[i] = np.nan
        
    return roots

# %%

@njit(fastmath = True)
def PREPARE_VECTORIZED(x : np.ndarray, y : np.ndarray, z : np.ndarray):
    times = SOLVE_VECTORIZED(x, z, wheel_speed_x)
    time_cos = np.cos(times)

    beta = wheel_radius * time_cos
    _wheel_y = wheel_speed_y * times
    delta = y - _wheel_y

    ret = (delta + _wheel_width2) / wheel_width
    ret[(~((z <= beta + EPS) & (beta * z >= -EPS))) | (np.abs(delta) > _wheel_width2) | (np.isnan(times))] = np.nan

    return time_cos, ret

# %%

TIMES_COS, KOEF = PREPARE_VECTORIZED(X, Y, plane.ravel())
active = np.where(~np.isnan(KOEF))[0]

projection = plane.ravel()[active]
X = X[active]
Y = Y[active]
TIMES_COS = TIMES_COS[active]
KOEF = KOEF[active]

# %%

@njit(fastmath = True)
def WORK(TIMES_COS : np.ndarray, KOEF : np.ndarray, projection : np.ndarray, zerno_graph : np.ndarray, zerno_pos : float, zerno_width : float):
    zerno_width /= wheel_width
    zerno_pos /= wheel_width

    k = (KOEF - (zerno_pos - zerno_width * 0.5)) / zerno_width

    mask = (k >= 0) & (k <= 1)
    idx = np.where(mask)[0]

    k = k[idx]

    step = 1 / (zerno_graph.size - 1)
    ki = (k / step).astype(np.int32)
    _k = k / step - ki
    height = (1 - _k) * zerno_graph[ki] + _k * zerno_graph[ki + 1] + wheel_radius

    z = TIMES_COS[idx] * height
    mask = z < projection[idx]
    idx = idx[mask]
    projection[idx] = z[mask]

    TIMES_COS[idx], KOEF[idx] = PREPARE_VECTORIZED(X[idx], Y[idx], projection[idx])

# %%

start = time()

for i in tqdm(range(zerno_count)):
    idx = np.random.randint(0, len(zerna))
    width = zerno_width()
    pos = np.random.uniform(0, wheel_width - width, (1, )).item() + width * 0.5
    WORK(TIMES_COS, KOEF, projection, zerna[idx] * zerno_height(), pos, width)

print(f'Took {time() - start} sec.')

plane[np.unravel_index(active, plane.shape)] = projection
plane += wheel_height

# %%

if save:
    from pandas import DataFrame
    DataFrame(plane).to_excel(save_filename, index = False, header = False)

# %%

m = display_max_z - display_min_z # max(plane_width, plane_height)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
fig.canvas.manager.set_window_title('Участок рабочей поверхности круга')

ax.set_xlabel('X (мкм)')
ax.set_ylabel('Y (мкм)')
ax.set_zlabel('Z (мкм)')
ax.set_box_aspect([plane_width / m, plane_height / m, 1])

# %%

X = np.linspace(0, plane_width, resolution_width).reshape(1, resolution_width).repeat(resolution_height, 0)
Y = np.linspace(0, plane_height, resolution_height).reshape(1, resolution_height).repeat(resolution_width, 0).T

ax.plot_surface(X, Y, plane, cmap = 'coolwarm_r')
# ax.plot(line[:, 0], line[:, 1], line[:, 2], color = 'r')

ax.set_xlim(0, plane_width)
ax.set_ylim(0, plane_height)
ax.set_zlim(display_min_z, display_max_z)

ax.set_xticks([0, plane_width // 2, plane_width])
ax.set_yticks([0, plane_height // 2, plane_height])
ax.set_zticks([display_min_z, 0, display_max_z])

plt.show()

# %%
