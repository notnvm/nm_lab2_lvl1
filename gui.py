import dearpygui.dearpygui as dpg
import dearpygui.demo as demo

class Data():
    test: bool = False
    solved: bool = False
    a,b,c,d,n,m = 0,0,0,0,0,0

def set_test(sender, data):
    Data.test = data
    if dpg.get_value(sender) == True:
        print(f'{sender}, active, {Data.test}')
    else:
        print(f'{sender}, inactive, {Data.test}')
        
def set_bound():
    Data.a = dpg.get_value(1)
    Data.b = dpg.get_value(2)
    Data.c = dpg.get_value(3)
    Data.d = dpg.get_value(4)
    Data.n = dpg.get_value(5)
    Data.m = dpg.get_value(6)
    print(f'a={Data.a}, b={Data.b}, c={Data.c}, d={Data.d}, n={Data.n}, m={Data.m}')
    
def solve_callback(sender, value, user_data):
    Data.solved = True
    dpg.set_value(user_data, f"{dpg.get_value(1)} < x < {dpg.get_value(2)}\tЧисло разбиений по x = {dpg.get_value(6)}\n{dpg.get_value(3)} < y < {dpg.get_value(4)}\tЧисло разбиений по y = {dpg.get_value(5)}")
    
    print(f'button pressed,{bool(Data.solved)}')
    print(user_data)
    
data = Data()  

dpg.create_context()
dpg.create_viewport(title="gui", resizable=False, width=1440, height=800)
dpg.setup_dearpygui()

with dpg.window(
    label="Example Window", 
    pos=[0,0], 
    width=dpg.get_viewport_width(), 
    height=dpg.get_viewport_height(), 
    no_title_bar=True,
    no_move=True ):
    
    with dpg.font_registry():
        with dpg.font(f'C:\\Windows\\Fonts\\arial.ttf', 16, tag="deft"):
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)
    dpg.bind_font("deft")
          
    dpg.draw_line([350, 30], [350, dpg.get_viewport_height()], color=(255,255,255,255))
    dpg.add_text("Ввод параметров", pos=[120,40])
    dpg.show_font_manager()
    dpg.add_checkbox(label="Test task", pos=[0, 60],  callback=set_test, default_value=Data.test, indent=1)
    dpg.add_input_int(label="= a", width=100, pos=[0,90], step=1, default_value=0, callback=set_bound, tag=1, on_enter=False, indent=1)
    dpg.add_input_int(label="= b", width=100, pos=[0,120], step=1, default_value=0, callback=set_bound, tag=2, on_enter=False, indent=1)
    dpg.add_input_int(label="= c", width=100, pos=[140,90], step=1, default_value=0, callback=set_bound, tag=3, on_enter=False)
    dpg.add_input_int(label="= d", width=100, pos=[140,120], step=1, default_value=0, callback=set_bound, tag=4, on_enter=False)
    dpg.add_input_int(label="= n", width=100, pos=[0,150], step=1, default_value=0, callback=set_bound, tag=5, on_enter=False, indent=1)
    dpg.add_input_int(label="= m", width=100, pos=[140,150], step=1, default_value=0, callback=set_bound, tag=6, on_enter=False)
    bounds_text=dpg.add_text(f"{data.a} < x < {data.b}\tЧисло разбиений по x = {data.m}\n{data.c} < y < {data.d}\tЧисло разбиений по y = {data.n}", pos=[0,210],indent=1, tag=8)
    dpg.add_button(label="Решить", width=320, pos=[0,180], tag=7, callback=solve_callback, indent=1, user_data=bounds_text)
    dpg.draw_line([0,260], [340,260])
    dpg.add_text("Справка", pos=[150,280])
    # demo.show_demo()
    with dpg.tab_bar():
        with dpg.tab(label="Тестовая задача", order_mode=True, tag=10):
             dpg.add_text("Тестовая", pos=[400,280])
             dpg.add_text("тест", pos=[0,310], indent=1)
        with dpg.tab(label="Основная задача",order_mode=True, tag=11):
             dpg.add_text("Основная", pos=[400,280])
             dpg.add_text("основа", pos=[0,310], indent=1)    

dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()


