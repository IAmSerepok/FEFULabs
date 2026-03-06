import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.widgets import Button, Slider
from configs import CONFIGURATIONS, get_config_names, get_config


class ThreeBodySimulation:
    def __init__(self):
        # Константы
        self.G = 1
        self.dt = 0.01
        self.speed_scale = 1
        self.view_scale = 3.0
        self.view_center_x = 0.0
        self.view_center_y = 0.0
        self.vector_edit_mode = False
        self.show_vectors = False
        self.trajectory_length = 100

        self.show_connecting_lines = False
        self.show_medians = False
        self.show_center_of_mass = False
        self.show_all_elements = False

        self.default_view_scale = 3.0
        self.default_view_center_x = 0.0
        self.default_view_center_y = 0.0

        self.anim_running = False
        self.anim = None

        self.dragging_body = None
        self.dragging_vector = None
        self.dragging_vector_start = None

        self.config_names = get_config_names()
        self.current_config_index = 0
        self.current_config_name = self.config_names[self.current_config_index]
        
        # Начальные условия
        self.reset_to_initial()
        
        # Создание фигуры и осей
        self.setup_plot()
        
    def reset_to_initial(self):
        config = get_config(self.current_config_name)
        
        self.r = [np.array(r).copy() for r in config["r"]]
        self.v = [np.array(v).copy() for v in config["v"]]
        self.m = config["m"].copy() if hasattr(config["m"], 'copy') else config["m"]
        
        self.trajectories = [[], [], []]
        self.current_step = 0
        self.state = self.create_state_vector()
        
    def reset_view_to_default(self):
        self.view_scale = self.default_view_scale
        self.view_center_x = self.default_view_center_x
        self.view_center_y = self.default_view_center_y
        self.update_view_limits()
        
    def create_state_vector(self):
        state = np.zeros(12)
        for i in range(3):
            state[2*i:2*i+2] = self.r[i]
            state[6+2*i:8+2*i] = self.v[i]
        return state
    
    def update_from_state(self):
        for i in range(3):
            self.r[i] = self.state[2*i:2*i+2].copy()
            self.v[i] = self.state[6+2*i:8+2*i].copy()
    
    def compute_midpoints(self):
        midpoints = [
            (self.r[0] + self.r[1]) / 2, 
            (self.r[1] + self.r[2]) / 2, 
            (self.r[2] + self.r[0]) / 2 
        ]
        return midpoints
    
    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(16, 9))
        plt.subplots_adjust(bottom=0.3, left=0.20, right=0.95, top=0.95)
        
        # Установка начального масштаба
        self.update_view_limits()
        
        self.ax.grid(True, alpha=0.3)
        
        # Цвета для тел
        self.colors = ['red', 'blue', 'green']
        
        # Инициализация графических элементов
        self.body_points = []
        self.velocity_arrows = []
        self.velocity_handles = []
        self.trajectory_lines = []
        self.trajectory_points = []

        self.connecting_lines = []
        self.median_lines = [] 
        self.midpoint_points = []
        self.median_intersection_point = None
        
        for i in range(3):
            # Точки тел
            point, = self.ax.plot([], [], 'o', color=self.colors[i], markersize=10, picker=8, zorder=10)
            self.body_points.append(point)
            
            # Векторы скорости
            arrow = self.ax.quiver([], [], [], [], color=self.colors[i], alpha=0.9, scale=1, scale_units='xy')
            self.velocity_arrows.append(arrow)
            
            # Ручки на концах векторов скорости
            handle, = self.ax.plot([], [], 's', color=self.colors[i], 
                                markersize=12, markerfacecolor='white',
                                markeredgecolor=self.colors[i], markeredgewidth=3,
                                picker=8, visible=False, zorder=10)
            self.velocity_handles.append(handle)
            
            # Линии траекторий
            line, = self.ax.plot([], [], color=self.colors[i], linewidth=2.0, alpha=0.6)
            self.trajectory_lines.append(line)
            
            # Точки траектории
            points, = self.ax.plot([], [], '.', color=self.colors[i], markersize=1.5, alpha=0.4)
            self.trajectory_points.append(points)
        
        # Линии между телами
        for i in range(3):
            line_conn, = self.ax.plot([], [], 'gray', linewidth=1.5, alpha=0.5, linestyle='--')
            self.connecting_lines.append(line_conn)
        
        # Медианы
        for i in range(3):
            line_med, = self.ax.plot([], [], 'purple', linewidth=2, alpha=0.6, linestyle=':', zorder=1)
            self.median_lines.append(line_med)
        
        # Точки середин сторон
        for i in range(3):
            midpoint_point, = self.ax.plot([], [], 'o', color='gray', markersize=6, alpha=0.7)
            self.midpoint_points.append(midpoint_point)
        
        # Точка центра масс
        self.median_intersection_point, = self.ax.plot([], [], 'mD', markersize=10,
                                                       markeredgecolor='white',
                                                       markeredgewidth=1)

        self.ax.set_title(f'Конфигурация: {self.current_config_name}', fontsize=14, fontweight='bold')
        
        # Создание кнопок и слайдеров
        self.create_controls()
        
        # Текстовая информация
        self.info_text = self.ax.text(-0.25, 0.5, '', transform=self.ax.transAxes,
                                    verticalalignment='top', fontsize=9,
                                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Текст с информацией об управлении видом
        self.view_info = self.ax.text(-0.25, 0.75, 'Управление видом:\n+ / - : масштаб\n←↑→↓ : перемещение',
                                     transform=self.ax.transAxes, fontsize=9,
                                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                                     verticalalignment='top')
        
        # Подключение событий
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Начальное обновление графика
        self.update_plot()
        
    def create_controls(self):
        # === КНОПКИ УПРАВЛЕНИЯ ===
        # Первый ряд кнопок
        ax_start = plt.axes([0.05, 0.20, 0.08, 0.03])
        ax_reset = plt.axes([0.14, 0.20, 0.08, 0.03])
        ax_clear = plt.axes([0.23, 0.20, 0.1, 0.03])
        
        self.btn_start = Button(ax_start, 'Старт')
        self.btn_reset = Button(ax_reset, 'Сброс')
        self.btn_clear = Button(ax_clear, 'Очистить траектории')
        
        self.btn_start.on_clicked(self.toggle_animation)
        self.btn_reset.on_clicked(self.reset_simulation)
        self.btn_clear.on_clicked(self.clear_trajectories)
        
        # Второй ряд кнопок
        ax_prev_config = plt.axes([0.05, 0.16, 0.08, 0.03])
        ax_next_config = plt.axes([0.14, 0.16, 0.08, 0.03])
        ax_config_info = plt.axes([0.23, 0.16, 0.1, 0.03])
        
        self.btn_prev_config = Button(ax_prev_config, '← Предыдущая')
        self.btn_next_config = Button(ax_next_config, 'Следующая →')
        
        # Кнопка с информацией о текущей конфигурации
        self.btn_config_info = Button(ax_config_info, f"{self.current_config_index + 1}/{len(self.config_names)}")
        self.btn_config_info.color = 'lightgray'
        self.btn_config_info.hovercolor = 'lightgray'
        
        self.btn_prev_config.on_clicked(self.prev_config)
        self.btn_next_config.on_clicked(self.next_config)
        
        # Третий ряд кнопок (режимы скорости)
        ax_vector_mode = plt.axes([0.05, 0.12, 0.12, 0.03])
        ax_show_vectors = plt.axes([0.18, 0.12, 0.12, 0.03]) 
        
        self.btn_vector_mode = Button(ax_vector_mode, 'Режим скорости: ВЫКЛ')
        self.btn_show_vectors = Button(ax_show_vectors, 'Векторы: ВЫКЛ')
        
        self.btn_vector_mode.on_clicked(self.toggle_vector_mode)
        self.btn_show_vectors.on_clicked(self.toggle_vectors)
        
        # === КНОПКИ ОТОБРАЖЕНИЯ ===
        ax_display_title = plt.axes([0.05, 0.07, 0.25, 0.02])
        ax_display_title.set_xticks([])
        ax_display_title.set_yticks([])
        ax_display_title.spines['top'].set_visible(False)
        ax_display_title.spines['right'].set_visible(False)
        ax_display_title.spines['bottom'].set_visible(False)
        ax_display_title.spines['left'].set_visible(False)

        ax_all_elements = plt.axes([0.05, 0.08, 0.25, 0.03])
        self.btn_all_elements = Button(ax_all_elements, 'Центр масс: ВЫКЛ')
        self.btn_all_elements.on_clicked(self.toggle_all_elements)
        
        # === СЛАЙДЕРЫ МАСС ===
        # Заголовок группы
        ax_mass_title = plt.axes([0.50, 0.22, 0.15, 0.02])
        ax_mass_title.set_xticks([])
        ax_mass_title.set_yticks([])
        ax_mass_title.text(0.5, 0.5, 'МАССЫ ТЕЛ', ha='center', va='center', 
                          fontsize=10, fontweight='bold', transform=ax_mass_title.transAxes)
        ax_mass_title.spines['top'].set_visible(False)
        ax_mass_title.spines['right'].set_visible(False)
        ax_mass_title.spines['bottom'].set_visible(False)
        ax_mass_title.spines['left'].set_visible(False)
        
        # Слайдер массы тела 1 (красное)
        ax_mass1 = plt.axes([0.55, 0.18, 0.12, 0.02])
        self.mass_slider1 = Slider(
            ax=ax_mass1,
            label='Тело 1',
            valmin=0.0,
            valmax=5.0,
            valinit=self.m[0],
            valstep=0.5,
            color='red'
        )
        self.mass_slider1.label.set_color('red')
        
        # Слайдер массы тела 2 (синее)
        ax_mass2 = plt.axes([0.55, 0.15, 0.12, 0.02])
        self.mass_slider2 = Slider(
            ax=ax_mass2,
            label='Тело 2',
            valmin=0.0,
            valmax=5.0,
            valinit=self.m[1],
            valstep=0.5,
            color='blue'
        )
        self.mass_slider2.label.set_color('blue')
        
        # Слайдер массы тела 3 (зеленое)
        ax_mass3 = plt.axes([0.55, 0.12, 0.12, 0.02])
        self.mass_slider3 = Slider(
            ax=ax_mass3,
            label='Тело 3',
            valmin=0.0,
            valmax=5.0,
            valinit=self.m[2],
            valstep=0.5,
            color='green'
        )
        self.mass_slider3.label.set_color('green')
        
        self.mass_slider1.on_changed(self.update_mass1)
        self.mass_slider2.on_changed(self.update_mass2)
        self.mass_slider3.on_changed(self.update_mass3)
        
        # === ПАРАМЕТРЫ СИМУЛЯЦИИ ===
        # Заголовок группы
        ax_params_title = plt.axes([0.75, 0.22, 0.15, 0.02])
        ax_params_title.set_xticks([])
        ax_params_title.set_yticks([])
        ax_params_title.text(0.5, 0.5, 'ПАРАМЕТРЫ', ha='center', va='center', 
                            fontsize=10, fontweight='bold', transform=ax_params_title.transAxes)
        ax_params_title.spines['top'].set_visible(False)
        ax_params_title.spines['right'].set_visible(False)
        ax_params_title.spines['bottom'].set_visible(False)
        ax_params_title.spines['left'].set_visible(False)
        
        # Слайдер скорости анимации
        ax_speed = plt.axes([0.80, 0.18, 0.12, 0.02])
        self.speed_slider = Slider(
            ax=ax_speed,
            label='Скорость',
            valmin=1,
            valmax=10,
            valinit=4,
            valstep=0.5
        )
        
        # Слайдер масштаба векторов
        ax_scale = plt.axes([0.80, 0.15, 0.12, 0.02])
        self.scale_slider = Slider(
            ax=ax_scale,
            label='Масштаб векторов',
            valmin=0.5,
            valmax=3,
            valinit=self.speed_scale,
            valstep=0.1
        )
        self.scale_slider.on_changed(self.update_vector_scale)
        
        # Длина траектории
        ax_trajectory = plt.axes([0.80, 0.12, 0.12, 0.02])
        self.trajectory_slider = Slider(
            ax=ax_trajectory,
            label='Длина траектории',
            valmin=100,
            valmax=2000,
            valinit=self.trajectory_length,
            valstep=50
        )
        self.trajectory_slider.on_changed(self.update_trajectory_length)
    
    def toggle_all_elements(self, event):
        self.show_all_elements = not self.show_all_elements

        self.show_connecting_lines = self.show_all_elements
        self.show_medians = self.show_all_elements
        self.show_center_of_mass = self.show_all_elements
        
        # Обновляем текст кнопки
        self.btn_all_elements.label.set_text(f'Центр масс: {"ВКЛ" if self.show_all_elements else "ВЫКЛ"}')
        
        self.update_plot()
    
    def on_key_press(self, event):
        if event.key == '=':
            self.view_scale *= 0.8
            self.update_view_limits()
            self.fig.canvas.draw_idle()
        elif event.key == '-':
            self.view_scale *= 1.25
            self.update_view_limits()
            self.fig.canvas.draw_idle()
        elif event.key == 'left':
            self.view_center_x -= self.view_scale * 0.5
            self.update_view_limits()
            self.fig.canvas.draw_idle()
        elif event.key == 'right':
            self.view_center_x += self.view_scale * 0.5
            self.update_view_limits()
            self.fig.canvas.draw_idle()
        elif event.key == 'up':
            self.view_center_y += self.view_scale * 0.5
            self.update_view_limits()
            self.fig.canvas.draw_idle()
        elif event.key == 'down':
            self.view_center_y -= self.view_scale * 0.5
            self.update_view_limits()
            self.fig.canvas.draw_idle()
    
    def update_mass1(self, val):
        if self.anim_running:
            self.toggle_animation(None)
        self.m[0] = val
        self.update_plot()
    
    def update_mass2(self, val):
        if self.anim_running:
            self.toggle_animation(None)
        self.m[1] = val
        self.update_plot()
    
    def update_mass3(self, val):
        if self.anim_running:
            self.toggle_animation(None)
        self.m[2] = val
        self.update_plot()
    
    def update_trajectory_length(self, val):
        self.trajectory_length = int(val)
        self.update_plot()
    
    def toggle_vectors(self, event):
        self.show_vectors = not self.show_vectors
        self.btn_show_vectors.label.set_text(f'Векторы: {"ВКЛ" if self.show_vectors else "ВЫКЛ"}')
        self.update_plot() 
        
    def update_view_limits(self):
        k = 25.5 / 10
        self.ax.set_xlim(self.view_center_x - self.view_scale * k, 
                        self.view_center_x + self.view_scale * k)
        self.ax.set_ylim(self.view_center_y - self.view_scale, 
                        self.view_center_y + self.view_scale)
    
    def prev_config(self, event):
        if self.anim_running:
            self.toggle_animation(None)
        
        self.current_config_index = (self.current_config_index - 1) % len(self.config_names)
        self.current_config_name = self.config_names[self.current_config_index]
        
        # Сбрасываем вид к стандартному
        self.reset_view_to_default()
        
        self.reset_to_initial()
        
        # Обновляем слайдеры масс
        self.mass_slider1.set_val(self.m[0])
        self.mass_slider2.set_val(self.m[1])
        self.mass_slider3.set_val(self.m[2])
        
        # Обновляем текст кнопки
        self.btn_config_info.label.set_text(f"{self.current_config_index + 1}/{len(self.config_names)}")
        self.ax.set_title(f'Конфигурация: {self.current_config_name}', fontsize=14, fontweight='bold')
        
        self.update_plot()
    
    def next_config(self, event):
        if self.anim_running:
            self.toggle_animation(None)
        
        self.current_config_index = (self.current_config_index + 1) % len(self.config_names)
        self.current_config_name = self.config_names[self.current_config_index]
        
        # Сбрасываем вид к стандартному
        self.reset_view_to_default()
        
        self.reset_to_initial()
        
        # Обновляем слайдеры масс
        self.mass_slider1.set_val(self.m[0])
        self.mass_slider2.set_val(self.m[1])
        self.mass_slider3.set_val(self.m[2])
        
        # Обновляем текст кнопки
        self.btn_config_info.label.set_text(f"{self.current_config_index + 1}/{len(self.config_names)}")
        self.ax.set_title(f'Конфигурация: {self.current_config_name}', fontsize=14, fontweight='bold')
        
        self.update_plot()
    
    def update_vector_scale(self, val):
        self.speed_scale = val
        self.update_plot()
    
    def toggle_vector_mode(self, event):
        self.vector_edit_mode = not self.vector_edit_mode
        self.btn_vector_mode.label.set_text(f'Режим скорости: {"ВКЛ" if self.vector_edit_mode else "ВЫКЛ"}')
        
        # Показываем/скрываем ручки векторов
        for handle in self.velocity_handles:
            handle.set_visible(self.vector_edit_mode)
        
        self.update_plot()
    
    def on_pick(self, event):
        if event.artist in self.body_points:
            if not self.vector_edit_mode:
                self.dragging_body = self.body_points.index(event.artist)
        
        # В режиме скорости
        elif event.artist in self.velocity_handles and self.vector_edit_mode:
            # Находим индекс вектора
            for i, handle in enumerate(self.velocity_handles):
                if event.artist == handle:
                    self.dragging_vector = i
                    # Запоминаем начальную позицию
                    self.dragging_vector_start = self.r[i] + self.v[i] * self.speed_scale
                    break
    
    def on_release(self, event):
        if self.dragging_body is not None:
            self.dragging_body = None
            if self.anim_running:
                self.toggle_animation(None)
        
        if self.dragging_vector is not None:
            self.dragging_vector = None
            self.dragging_vector_start = None
            if self.anim_running:
                self.toggle_animation(None)
    
    def on_motion(self, event):
        if event.inaxes != self.ax:
            return
        
        # Перетаскивание тела
        if self.dragging_body is not None and not self.vector_edit_mode:
            if event.xdata is not None and event.ydata is not None:
                # Обновление позиции тела
                self.r[self.dragging_body] = np.array([event.xdata, event.ydata])
                
                # Обновление вектора состояния
                self.state[2*self.dragging_body:2*self.dragging_body+2] = self.r[self.dragging_body]
                
                # Очистка траекторий при ручном перемещении
                self.clear_trajectories(None)
                
                self.update_plot()
        
        # Изменение вектора скорости
        elif self.dragging_vector is not None and self.vector_edit_mode:
            if event.xdata is not None and event.ydata is not None:
                i = self.dragging_vector
                
                # Вычисляем новую скорость на основе позиции курсора
                new_vector = np.array([event.xdata, event.ydata]) - self.r[i]
                
                # Масштабируем обратно, чтобы получить реальную скорость
                if self.speed_scale > 0:
                    self.v[i] = new_vector / self.speed_scale
                
                # Обновление вектора состояния
                self.state[6+2*i:8+2*i] = self.v[i]
                
                self.clear_trajectories(None)
                
                self.update_plot()
    
    def compute_accelerations(self, r):
        a = [np.array([0.0, 0.0]) for _ in range(3)]
        
        for i in range(3):
            for j in range(3):
                if i != j:
                    r_ij = r[j] - r[i]
                    dist = np.linalg.norm(r_ij)
                    if dist > 0.05:  # ! Защита от схлопывания
                        a[i] += self.G * self.m[j] * r_ij / (dist**3 + 1e-8)
        
        return a
    
    def system_derivatives(self, state): 
        r_current = [state[0:2], state[2:4], state[4:6]]
        v_current = [state[6:8], state[8:10], state[10:12]]
        
        a_current = self.compute_accelerations(r_current)
        
        derivatives = np.zeros(12)
        derivatives[0:2] = v_current[0]
        derivatives[2:4] = v_current[1]
        derivatives[4:6] = v_current[2]
        derivatives[6:8] = a_current[0]
        derivatives[8:10] = a_current[1]
        derivatives[10:12] = a_current[2]
        
        return derivatives
    
    def rk4_step(self, state, dt):
        k1 = self.system_derivatives(state)
        k2 = self.system_derivatives(state + 0.5 * dt * k1)
        k3 = self.system_derivatives(state + 0.5 * dt * k2)
        k4 = self.system_derivatives(state + dt * k3)
        
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def update_plot(self):
        # Обновление позиций тел
        for i in range(3):
            self.body_points[i].set_data([self.r[i][0]], [self.r[i][1]])
            
            # Вычисляем конец вектора скорости
            arrow_end = self.r[i] + self.v[i] * self.speed_scale
            
            # Обновление векторов скорости
            self.velocity_arrows[i].remove()
            if self.show_vectors: 
                self.velocity_arrows[i] = self.ax.quiver(
                    self.r[i][0], self.r[i][1], 
                    self.v[i][0] * self.speed_scale, self.v[i][1] * self.speed_scale,
                    color=self.colors[i], alpha=0.9, scale=1, scale_units='xy', 
                    width=0.005, 
                    headlength=2, 
                    headaxislength=1.5, 
                    minshaft=1,  
                    picker=True, 
                    pickradius=3)
            else:
                self.velocity_arrows[i] = self.ax.quiver(
                    [], [], [], [],
                    color=self.colors[i], alpha=0.0, scale=1, scale_units='xy')
            
            # Обновление ручек на концах векторов
            if self.vector_edit_mode and self.show_vectors: 
                self.velocity_handles[i].set_data([arrow_end[0]], [arrow_end[1]])
                self.velocity_handles[i].set_visible(True)
            else:
                self.velocity_handles[i].set_visible(False)
        
        # Обновление линий между телами
        if self.show_connecting_lines:
            pairs = [(0, 1), (1, 2), (2, 0)]
            for idx, (i, j) in enumerate(pairs):
                self.connecting_lines[idx].set_data(
                    [self.r[i][0], self.r[j][0]],
                    [self.r[i][1], self.r[j][1]]
                )
                self.connecting_lines[idx].set_visible(True)
        else:
            for line in self.connecting_lines:
                line.set_visible(False)
        
        # Обновление медиан
        if self.show_medians:
            # Вычисляем середины сторон
            midpoints = self.compute_midpoints()

            medians = [(0, 1), (1, 2), (2, 0)] 
            
            for idx, (vertex, mid_idx) in enumerate(medians):
                self.median_lines[idx].set_data(
                    [self.r[vertex][0], midpoints[mid_idx][0]],
                    [self.r[vertex][1], midpoints[mid_idx][1]]
                )
                self.median_lines[idx].set_visible(True)
            
            # Показываем точки середин сторон
            for idx, midpoint in enumerate(midpoints):
                self.midpoint_points[idx].set_data([midpoint[0]], [midpoint[1]])
                self.midpoint_points[idx].set_visible(True)

            median_intersection = np.mean(self.r, axis=0)
            self.median_intersection_point.set_data([median_intersection[0]], [median_intersection[1]])
            self.median_intersection_point.set_visible(True)
        else:
            for line in self.median_lines:
                line.set_visible(False)
            for point in self.midpoint_points:
                point.set_visible(False)
            self.median_intersection_point.set_visible(False)
        
        # Обновление траекторий с учетом текущей длины
        for i in range(3):
            if len(self.trajectories[i]) > 0:
                # Ограничиваем длину траектории
                if len(self.trajectories[i]) > self.trajectory_length:
                    self.trajectories[i] = self.trajectories[i][-self.trajectory_length:]
                
                traj = np.array(self.trajectories[i])
                self.trajectory_lines[i].set_data(traj[:, 0], traj[:, 1])
                self.trajectory_points[i].set_data(traj[:, 0], traj[:, 1])
            else:
                self.trajectory_lines[i].set_data([], [])
                self.trajectory_points[i].set_data([], [])
        
        info = f'Время: {self.current_step * self.dt:.2f} с\n'
        info += f'Шаг: {self.current_step}\n'
        info += '\n'
        for i, j in [(1, 2), (2, 3), (1, 3)]:
            r_ij = np.sum((self.r[i-1] - self.r[j-1]) ** 2)
            info += f'r{i}{j}^2 = {r_ij:.2f}' + '\n'
        info += '\n'
        for i in range(3):
            info += f'Тело {i+1} ({self.colors[i]}):\n'
            info += f'  r=({self.r[i][0]:.2f}, {self.r[i][1]:.2f})\n'
            speed = np.linalg.norm(self.v[i])
            info += f'  v=({self.v[i][0]:.2f}, {self.v[i][1]:.2f}) |v|={speed:.2f}\n'
        
        self.info_text.set_text(info)
        
        # Принудительно обновляем фигуру
        self.fig.canvas.draw_idle()
    
    def toggle_animation(self, event):
        if self.anim_running:
            self.anim_running = False
            self.btn_start.label.set_text('Старт')
            if self.anim:
                self.anim.event_source.stop()
        else:
            self.anim_running = True
            self.btn_start.label.set_text('Стоп')
            self.run_animation()
    
    def run_animation(self):
        def animate(frame):
            if not self.anim_running:
                return
            
            # Несколько шагов за кадр в зависимости от скорости
            steps_per_frame = int(self.speed_slider.val)
            for _ in range(steps_per_frame):
                # Сохранение позиций
                for i in range(3):
                    self.trajectories[i].append(self.r[i].copy())
                
                # Шаг интегрирования
                self.state = self.rk4_step(self.state, self.dt)
                self.update_from_state()
                self.current_step += 1
            
            self.update_plot()
        
        self.anim = animation.FuncAnimation(self.fig, animate, interval=50, blit=False)
        plt.draw()
    
    def reset_simulation(self, event):
        self.reset_to_initial()
        # Сбрасываем вид к стандартному
        self.reset_view_to_default()
        # Обновляем слайдеры масс
        self.mass_slider1.set_val(self.m[0])
        self.mass_slider2.set_val(self.m[1])
        self.mass_slider3.set_val(self.m[2])
        self.update_plot()
    
    def clear_trajectories(self, event):
        self.trajectories = [[], [], []]
        self.current_step = 0
        self.update_plot()


# Запуск приложения
if __name__ == "__main__":
    sim = ThreeBodySimulation()
    plt.show()

# TODO тогл для беск массы
