extends Control

# Konstanta path scene untuk menghindari typo & mempermudah refactor
const TRYON_SCENE_PATH := "res://Scenes/TryOn.tscn"
const CREDIT_SCENE_PATH := "res://Scenes/Credit.tscn"
const BANTUAN_SCENE_PATH := "res://Scenes/Bantuan.tscn"

func _ready() -> void:
	# Hubungkan tombol-tombol dengan fungsi event handler masing-masing
	$Panel/ButtonContainer/Start.pressed.connect(_on_start_pressed)
	$Panel/ButtonContainer/Credit.pressed.connect(_on_credit_pressed)
	$Panel/ButtonContainer/Help.pressed.connect(_on_help_pressed)
	$Panel/ButtonContainer/Quit.pressed.connect(_on_quit_pressed)

	if SoundManager and not SoundManager.get_node("MusicPlayer").is_playing():
		SoundManager.get_node("MusicPlayer").stop()


# Fungsi saat tombol "Start" ditekan - navigasi ke scene utama
func _on_start_pressed() -> void:
	if SoundManager and not SoundManager.get_node("MusicPlayer").is_playing():
		SoundManager.get_node("MusicPlayer").play()
	get_tree().change_scene_to_file(TRYON_SCENE_PATH)


# Fungsi saat tombol "Credit" ditekan - navigasi ke scene kredit
func _on_credit_pressed() -> void:
	get_tree().change_scene_to_file(CREDIT_SCENE_PATH)


# Fungsi saat tombol "Help" ditekan - navigasi ke scene help
func _on_help_pressed() -> void:
	get_tree().change_scene_to_file(BANTUAN_SCENE_PATH)


# Fungsi saat tombol "Quit" (Keluar) ditekan - keluar dari game
func _on_quit_pressed() -> void:
	get_tree().quit()
