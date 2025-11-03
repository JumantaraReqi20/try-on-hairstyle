extends Node

# Fungsi umum untuk balik ke Main Menu
func go_back_to_main(is_connected: bool = false, disconnect_func: Callable = Callable()):
	if is_connected and disconnect_func.is_valid():
		disconnect_func.call()
	
	get_tree().change_scene_to_file("res://Scenes/MainMenu.tscn")


# Fungsi umum untuk keluar dari game
func quit_game():
	get_tree().quit()
