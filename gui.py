import dearpygui.dearpygui as dpg
import dearpygui.demo as demo

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg

import numpy as np

import solution as sol
from solution import dv 

#! #####################################
# @TODO: #*Проверить с неодинаковыми n,m
#! #####################################

class TaskSolution():
    v_num = np.zeros((dv.n + 1, dv.m + 1))
    u_exact = np.zeros((dv.n + 1, dv.m + 1))
    diff = np.zeros((dv.n + 1, dv.m + 1))
    
    v12_num = np.zeros((dv.n*2 + 1, dv.m*2 + 1))
    diffv12 = np.zeros((dv.n + 1, dv.m + 1))
    s_count, eps_max, omega = 0,0,0
    s_count2, eps_max2, omega2 = 0,0,0
    
task_sol = TaskSolution()
class Callbacks:
    test_tab = True
    use_optimal_omega = False
    
    def update_help(self):
        if menu.cb.test_tab:
            dpg.set_value(item='help_omega', value=f"Параметр метода omega: {task_sol.omega}")
            dpg.set_value(item='s_counter', value=f"Итераций затрачено на решение: {task_sol.s_count}")
            dpg.set_value(item='eps_max_solved', value=f"Достигнутая точность eps: {task_sol.eps_max}")
        else:
            dpg.set_value(item='help_omega', value=f"Параметр метода omega: {task_sol.omega2}")
            dpg.set_value(item='s_counter', value=f"Итераций затрачено на решение: {task_sol.s_count2}")
            dpg.set_value(item='eps_max_solved', value=f"Достигнутая точность eps: {task_sol.eps_max2}")
        
    def radio_butn_cb(self,sender,data):
        print(dpg.get_value('rb1'))
        if dpg.get_value('rb1') == menu.rb_item_list[0]:
            menu.create_table(dv.m + 1, dv.n + 1, task_sol.v_num)
        if dpg.get_value('rb1') == menu.rb_item_list[1]:
            menu.create_table(dv.m + 1, dv.n + 1, task_sol.u_exact)
        if dpg.get_value('rb1') == menu.rb_item_list[2]:
            menu.create_table(dv.m + 1, dv.n + 1, task_sol.diff)
            
    def radio_butn_cb_main(self,sender,data):
        print(dpg.get_value('rb_main'))
        if dpg.get_value('rb_main') == menu.rb_item_list_main[0]:
            menu.create_table(dv.m + 1, dv.n + 1, task_sol.v_num)
        if dpg.get_value('rb_main') == menu.rb_item_list_main[1]:
            menu.create_table(dv.m*2+1, dv.n*2+1, task_sol.v12_num)
        if dpg.get_value('rb_main') == menu.rb_item_list_main[2]:
            menu.create_table(dv.m + 1, dv.n + 1, task_sol.diffv12)
    
    def set_test(self, sender, data):
        dv.main_task = data
        if dpg.get_value(sender) == True:
            print(f'{sender}, active, {dv.main_task}')
        else:
            print(f'{sender}, inactive, {dv.main_task}')
            
    def use_omega(self):
        self.use_optimal_omega = not self.use_optimal_omega
        print(f'optimal omega={bool(self.use_optimal_omega)}')

    def tb_callback(self):
        dv.main_task = not dv.main_task
        self.test_tab = not self.test_tab
        self.update_help()

    def set_bound(self):
        dv.a = dpg.get_value(1)
        dv.b = dpg.get_value(2)
        dv.c = dpg.get_value(3)
        dv.d = dpg.get_value(4)
        dv.n = dpg.get_value(5)
        dv.m = dpg.get_value(6)
        dv.eps = dpg.get_value(7)
        dv.nmax = dpg.get_value(8)
        print(f'a={dv.a}, b={dv.b}, c={dv.c}, d={dv.d}, n={dv.n}, m={dv.m}, eps={dv.eps:e}, nmax={dv.nmax}, main={dv.main_task}')
           
    def solve_callback(self, sender, value, user_data):
        dv.solved = True
        dpg.set_value(user_data, f"{dpg.get_value(1)} < x < {dpg.get_value(2)}\tЧисло разбиений по x = {dpg.get_value(6)}\n{dpg.get_value(3)} < y < {dpg.get_value(4)}\tЧисло разбиений по y = {dpg.get_value(5)}")
        
        
        x = np.linspace(dv.a, dv.b, dv.n+1)
        y = np.linspace(dv.c, dv.d, dv.m+1)
        v = np.zeros((dv.n + 1, dv.m + 1))
        sol.fill_bounds_v(v, x, y, dv.n, dv.m)
        
        # v_sol, s_count, eps_max, omega = 0,0,0,0
        if self.use_optimal_omega:
            omega = sol.optimal_w(sol.fill_matrix(dv.n, dv.m))
            omega2 = sol.optimal_w(sol.fill_matrix(dv.n*2, dv.m*2))
            if not dv.main_task:
                task_sol.v_num, task_sol.s_count, task_sol.eps_max, task_sol.omega = sol.upper_relaxation(v, dv.n, dv.m, omega) # type: ignore
            else:
                task_sol.v_num, task_sol.s_count2, task_sol.eps_max2, task_sol.omega2 = sol.upper_relaxation(v, dv.n, dv.m, omega) # type: ignore 
                #* find solution, meshgrid2 = 1/2meshgrid1
                x12 = np.linspace(dv.a, dv.b, dv.n*2+1)
                y12 = np.linspace(dv.c, dv.d, dv.m*2+1)
                v12 = np.zeros((dv.n*2 + 1, dv.m*2 + 1))
                sol.fill_bounds_v(v12, x12, y12, dv.n*2, dv.m*2)
                task_sol.v12_num, sc2, epm2, omega2 = sol.upper_relaxation(v12, dv.n*2, dv.m*2, omega2) # type: ignore 
                print(f'\n\n\nv_num12=\n{task_sol.v12_num}\n\n\n')
        else:
            if not dv.main_task:
                task_sol.v_num, task_sol.s_count, task_sol.eps_max, task_sol.omega = sol.upper_relaxation(v, n=dv.n, m=dv.m) # type: ignore
            else:
                task_sol.v_num, task_sol.s_count2, task_sol.eps_max2, task_sol.omega2 = sol.upper_relaxation(v, n=dv.n, m=dv.m) # type: ignore
                #* find solution, meshgrid2 = 1/2meshgrid1
                x12 = np.linspace(dv.a, dv.b, dv.n*2+1)
                y12 = np.linspace(dv.c, dv.d, dv.m*2+1)
                v12 = np.zeros((dv.n*2 + 1, dv.m*2 + 1))
                sol.fill_bounds_v(v12, x12, y12, dv.n*2, dv.m*2)
                task_sol.v12_num, sc2, epm2, omega2 = sol.upper_relaxation(v12, dv.n*2, dv.m*2) # type: ignore 
                print(f'\n\n\nv_num12=\n{task_sol.v12_num}\n\n\n')
            
        task_sol.u_exact=sol.find_exact_solution(x, y, dv.n, dv.m)
        task_sol.diff = np.fabs(task_sol.u_exact - task_sol.v_num) # type: ignore
        v12_num_sliced = task_sol.v12_num[::2, ::2]
        task_sol.diffv12 = np.fabs(task_sol.v_num - v12_num_sliced)
        
        menu.create_table(dv.m + 1, dv.n + 1, task_sol.v_num[:][::-1])     
        self.update_help()
        
        plot_img = menu.setup_plot(x,y,task_sol.v_num)
        if self.test_tab:
            dpg.delete_item("texture_id")
            dpg.delete_item("plot_texture")
            dpg.add_raw_texture( 640, 480, plot_img, format=dpg.mvFormat_Float_rgba, tag="texture_id",parent='tex_reg') # type: ignore
            dpg.add_image("texture_id", pos=[int((menu.width-menu.ofx)/4.5),menu.height-500], tag='plot_texture', parent='test_tab') 
            
        if not self.test_tab:
            dpg.delete_item("texture_id_main")
            dpg.delete_item("plot_texture_main")
            dpg.add_raw_texture( 640, 480, plot_img, format=dpg.mvFormat_Float_rgba, tag="texture_id_main",parent='tex_reg') # type: ignore
            dpg.add_image("texture_id_main", pos=[int((menu.width-menu.ofx)/4.5),menu.height-500], tag='plot_texture_main', parent='main_tab') 
        
        print(f'button pressed, solved={bool(dv.solved)}\nx={x}, y={y}') 
        print(f'test_task={bool(not dv.main_task)}\n') 
        print(f'\ns={task_sol.s_count}, eps_max={np.around(task_sol.eps_max,5)},\nv_sol={task_sol.v_num}')
        
        # print(np.around(v[::-1], 3))
        # for row in np.around(v, 3):
        #     print(*row)
        # print('\n') #! правая сторона моя - верх в проге, левая моя - низ прога, верх мой - левая прога, низ мой - правая прога( тут с обратным знаком)??? 
        #* В программе это строка, у меня - столбец
                     
class Menu:
    title: str
    width: int
    height: int
    pos: list = [0,0]
    ofx: int = 410
    ofy: int = 70
    cb = Callbacks()
    rb_item_list = ["Численноe решениe", "Точноe решениe", "Разность решений"]
    rb_item_list_main = ["Численноe решениe", "Численное решение на сетке с половинным шагом", "Разность решений"]
     
    def __init__(self, t, w, h, p = pos):
        self.title = t
        self.width = w
        self.height = h
        self.pos = p
        
    def setup_font(self, path_to_font: str, sz: int, tag: str):
        with dpg.font_registry():
            with dpg.font(path_to_font, sz, tag=tag):
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)
        dpg.bind_font(tag)
        
    def setup_inputs(self, offset_x=ofx, offset_y=ofy):
         dpg.add_input_int(label="= a", width=130, pos=[self.width - offset_x,offset_y], step=1, default_value=0, callback=self.cb.set_bound, tag=1, on_enter=False)
         dpg.add_input_int(label="= b", width=130, pos=[self.width - offset_x,int(offset_y*1.5)], step=1, default_value=0, callback=self.cb.set_bound, tag=2, on_enter=False)
         dpg.add_input_int(label="= c", width=130, pos=[self.width - offset_x//2,offset_y], step=1, default_value=0, callback=self.cb.set_bound, tag=3, on_enter=False)
         dpg.add_input_int(label="= d", width=130, pos=[self.width - offset_x//2,int(offset_y*1.5)], step=1, default_value=0, callback=self.cb.set_bound, tag=4, on_enter=False)
         dpg.add_input_int(label="= n", width=130, pos=[self.width - offset_x,offset_y*2], step=1, default_value=0, callback=self.cb.set_bound, tag=5, on_enter=False)
         dpg.add_input_int(label="= m", width=130, pos=[self.width - offset_x//2,offset_y*2], step=1, default_value=0, callback=self.cb.set_bound, tag=6, on_enter=False)
         #tag = 9
         dpg.add_input_float(label="= eps", width=130, pos=[self.width - offset_x,int(offset_y*2.5)], step=1, default_value=dv.eps, callback=self.cb.set_bound, tag=7, on_enter=False, format='%.1e')
         #rag = 12
         dpg.add_input_int(label="= Nmax", width=130, pos=[self.width - offset_x//2,int(offset_y*2.5)], step=1, default_value=dv.nmax, callback=self.cb.set_bound, tag=8, on_enter=False)
    
    def create_table(self, num_cols, num_rows, value):
        if not dv.main_task:
            dpg.delete_item('table100', children_only=True)  
            for j in range(num_cols):
                dpg.add_table_column(label='V(i,j)', parent='table100')
            for i in range(num_rows):
                with dpg.table_row(parent='table100'):
                    for j in range(num_cols):
                        dpg.add_text(f"{np.around(value[i][j],3)}")
        if dv.main_task:
            dpg.delete_item('table100_main', children_only=True)  
            for j in range(num_cols):
                dpg.add_table_column(label='V(i,j)', parent='table100_main')
            for i in range(num_rows):
                with dpg.table_row(parent='table100_main'):
                    for j in range(num_cols):
                        dpg.add_text(f"{np.around(value[i][j],3)}")
                            
    def setup_plot(self, x=np.zeros(2), y=np.zeros(2), z=np.zeros((2,2))):
        # facecolor='#252526'
        fig = plt.figure(facecolor='#252526')
        sp = fig.add_subplot(projection='3d')
        sp.set_title('График решения', color='white')
        x, y = np.meshgrid(x, y)
        z.reshape(x.shape)
        sp.plot_surface(x, y, z, color='skyblue')

        canvas = FigureCanvasAgg(fig)
        ax = fig.gca()
        ax.set_facecolor(color='#252526')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white') # type: ignore
        # ax.set_zlim(-1,1)
        # ax.set_xlabel('$x$', fontsize=17, color='white')
        # ax.set_ylabel('$y$', fontsize=17, color='white')
        # ax.set_zlabel('$V(x,y)$', fontsize=17, color='white') # type: ignore
        canvas.draw()
        buf = canvas.buffer_rgba()
        image = np.asarray(buf)
        image = image.astype(np.float32) / 255
        # plt.savefig('G://dev//nm_level_1//img.png')

        return image                 
                            
    def setup_help(self):
        dpg.add_text("- Применялся метод верхней релаксации", pos=[self.width - self.ofx,340+30])
        dpg.add_text("- Начальное приближение: нулевое", pos=[self.width - self.ofx,370+30])
        dpg.add_text("Параметр метода omega: ", pos=[self.width - self.ofx,400+30], tag='help_omega')
        dpg.add_text("Итераций затрачено на решение: ", pos=[self.width - self.ofx,430+30], tag='s_counter')
        dpg.add_text("Достигнутая точность eps: ", pos=[self.width - self.ofx,460+30], tag='eps_max_solved')
    
dpg.create_context()
dpg.create_viewport(title="gui", resizable=False, width=1680, height=960)
dpg.setup_dearpygui()
    
menu = Menu('gui', dpg.get_viewport_width(), dpg.get_viewport_height())    

menu.setup_font('C:\\Windows\\Fonts\\arial.ttf', 16, tag='arial')    
    
with dpg.window(
    label=menu.title, 
    pos=menu.pos, 
    width=menu.width, 
    height=menu.height, 
    no_title_bar=True,
    no_move=True,
    tag='main_window_1'):

    dpg.draw_line([menu.width - menu.ofx - 20, menu.ofy//2], [menu.width - menu.ofx - 20, menu.height - int(menu.ofy/1.25)], color=(255,255,255,255))
    dpg.add_text("Ввод параметров", pos=[menu.width - int(menu.ofx/1.5),int(menu.ofy/1.75)])
    menu.setup_inputs()
    
    bounds_text=dpg.add_text(f"{dv.a} < x < {dv.b}\tЧисло разбиений по x = {dv.m}\n{dv.c} < y < {dv.d}\tЧисло разбиений по y = {dv.n}", pos=[menu.width - menu.ofx,int(menu.ofy*3.5)+30])
    
    dpg.add_checkbox(label='Использовать оптимальное omega', pos=[menu.width - menu.ofx,menu.ofy*3], callback=menu.cb.use_omega, tag='use_omega_cbox')
    dpg.add_button(label="Решить", width=390, pos=[menu.width - menu.ofx,menu.ofy*3+30], callback=menu.cb.solve_callback, user_data=bounds_text)
    
    dpg.draw_line([menu.width - menu.ofx - 10,menu.ofy*4+30], [menu.width - 30,menu.ofy*4+30])
    dpg.add_text("Справка", pos=[menu.width-int(menu.ofx/1.7),int(menu.ofy*4.35)+30])
    demo.show_demo()
    menu.setup_help()
    
    with dpg.tab_bar(callback=menu.cb.tb_callback):
        with dpg.tab(label="Тестовая задача", order_mode=True, tag='test_tab'):
            
             dpg.add_table(header_row=True, row_background=True, borders_innerV=True, borders_innerH=True, borders_outerH=True, borders_outerV=True, resizable=True, no_host_extendX=True, width=menu.width-440, height=menu.height-550, tag='table100', scrollY=True)        
             menu.create_table(1,1,np.zeros((3,3)))
             
            #  menu.setup_help()
             dpg.add_texture_registry(tag='tex_reg')
             dpg.add_raw_texture( 640, 480, menu.setup_plot(), format=dpg.mvFormat_Float_rgba, tag="texture_id",parent='tex_reg') # type: ignore
             dpg.add_image("texture_id", pos=[int((menu.width-menu.ofx)/4.5),menu.height-500], tag='plot_texture')  
            #  dpg.add_text("Достигнутая точность eps: ", pos=[self.width - self.ofx,460+30], tag='eps_max_solved')
             dpg.add_text('Таблица: ', pos=[menu.width - menu.ofx,600])
             dpg.add_radio_button(items=menu.rb_item_list, horizontal=False, pos=[menu.width - menu.ofx,630], callback=menu.cb.radio_butn_cb, tag='rb1')
             
        with dpg.tab(label="Основная задача", order_mode=True, tag='main_tab'):
            
            #  dpg.add_text("Основная", pos=[400,280])  
             dpg.add_table(header_row=True, row_background=True, borders_innerV=True, borders_innerH=True, borders_outerH=True, borders_outerV=True, resizable=True, no_host_extendX=True, width=menu.width-440, height=menu.height-550, tag='table100_main', scrollY=True)
             
             dpg.add_raw_texture( 640, 480, menu.setup_plot(), format=dpg.mvFormat_Float_rgba, tag="texture_id_main",parent='tex_reg') # type: ignore
             dpg.add_image("texture_id_main", pos=[int((menu.width-menu.ofx)/4.5),menu.height-500], tag='plot_texture_main') 
             dpg.add_text('Таблица: ', pos=[menu.width - menu.ofx,600])
             dpg.add_radio_button(items=menu.rb_item_list_main, horizontal=False, pos=[menu.width - menu.ofx,630], callback=menu.cb.radio_butn_cb_main, tag='rb_main')
             


dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
