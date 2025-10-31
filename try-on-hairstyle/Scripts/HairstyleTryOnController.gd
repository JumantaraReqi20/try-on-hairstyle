extends Control

@onready var loading_overlay = $LoadingOverlay
@onready var face_frame = $MainContainer/CameraContainer/WebcamContainer/WebcamFeed/DetectionOverlay/FaceFrame
@onready var webcam_feed = $MainContainer/CameraContainer/WebcamContainer/WebcamFeed
@onready var camera_status_label = $MainContainer/CameraContainer/WebcamContainer/WebcamFeed/CameraStatusLabel
@onready var loading_spinner = $LoadingOverlay/LoadingContainer/LoadingSpinner

# Webcam Manager - akan di-load secara manual
var webcam_manager: Node

var spinner_rotation: float = 0.0
var webcam_frames_received: int = 0
	print("=== EthnicityDetectionController._ready() ===")
	print("Scene tree ready, setting up webcam...")
	
	# Verifikasi node path dulu
	print("Verifying node paths...")
	print("webcam_feed path exists: ", has_node("MainContainer/CameraContainer/WebcamContainer/WebcamFeed"))
	print("camera_status_label path exists: ", has_node("MainContainer/CameraContainer/WebcamContainer/WebcamFeed/CameraStatusLabel"))
	
	setup_webcam_manager()
	setup_timers()
	reset_ui()
	setup_loading_spinner()

func setup_webcam_manager():
	"""Setup WebcamManager untuk real webcam"""
	print("=== Setting up WebcamManager ===")
	
	# Verifikasi node tersedia
	if not webcam_feed:
		print("ERROR: webcam_feed node not found!")
		return
	
	if not camera_status_label:
		print("ERROR: camera_status_label node not found!")
		return
	
	# Setup placeholder image dulu
	setup_webcam_placeholder()
	
	# Load WebcamManager script dengan path yang benar
	var webcam_script = load("res://Scenes/EthnicityDetection/WebcamClient/WebcamManager.gd")
	if webcam_script == null:
		print("Error: Could not load WebcamManager.gd")
		camera_status_label.text = "Error: WebcamManager tidak ditemukan"
		camera_status_label.modulate = Color(1, 0, 0, 0.8)
		return
	
	print("Creating WebcamManager instance...")
	webcam_manager = webcam_script.new()
	add_child(webcam_manager)
	
	# Connect signals dengan error handling
	print("Connecting signals...")
	if webcam_manager.has_signal("frame_received"):
		webcam_manager.frame_received.connect(_on_webcam_frame_received)
		print("✅ frame_received signal connected")
	else:
		print("❌ frame_received signal not found")
	
	if webcam_manager.has_signal("connection_changed"):
		webcam_manager.connection_changed.connect(_on_webcam_connection_changed)
		print("✅ connection_changed signal connected")
	else:
		print("❌ connection_changed signal not found")
	
	if webcam_manager.has_signal("error_message"):
		webcam_manager.error_message.connect(_on_webcam_error)
		print("✅ error_message signal connected")
	else:
		print("❌ error_message signal not found")
	
	# Update status
	camera_status_label.text = "Mencoba koneksi ke webcam server..."
	camera_status_label.modulate = Color(1, 1, 0, 0.8)
	
	# Coba koneksi ke webcam server
	print("Attempting to connect to webcam server...")
	webcam_manager.connect_to_webcam_server()
	print("WebcamManager setup complete")

func setup_webcam_placeholder():
	"""Buat placeholder image untuk webcam"""
	var placeholder_image = Image.create(640, 480, false, Image.FORMAT_RGBA8)
	placeholder_image.fill(Color(0.2, 0.2, 0.3, 1.0))
	
	# Buat border
	for x in range(640):
		for y in range(10):
			placeholder_image.set_pixel(x, y, Color(0.4, 0.4, 0.5, 1.0))
			placeholder_image.set_pixel(x, 479-y, Color(0.4, 0.4, 0.5, 1.0))
	
	for y in range(480):
		for x in range(10):
			placeholder_image.set_pixel(x, y, Color(0.4, 0.4, 0.5, 1.0))
			placeholder_image.set_pixel(639-x, y, Color(0.4, 0.4, 0.5, 1.0))
	
	var placeholder_texture = ImageTexture.new()
	placeholder_texture.set_image(placeholder_image)
	webcam_feed.texture = placeholder_texture

func _on_webcam_frame_received(texture: ImageTexture):
	"""Callback ketika frame webcam diterima"""
	print("Frame received! Size: ", texture.get_size())
	
	if not webcam_feed:
		print("ERROR: webcam_feed node is null!")
		return
	
	webcam_feed.texture = texture
	webcam_frames_received += 1
	
	# Update status untuk menunjukkan webcam aktif
	if webcam_frames_received == 1:
		print("First frame received, updating status...")
		camera_status_label.text = "Webcam aktif - Frame: " + str(webcam_frames_received)
		camera_status_label.modulate = Color(0, 1, 0, 0.8)
		
		# Hide status label setelah beberapa saat
		var hide_timer = Timer.new()
		hide_timer.wait_time = 3.0
		hide_timer.one_shot = true
		hide_timer.timeout.connect(func(): 
			if camera_status_label:
				camera_status_label.visible = false
		)
		add_child(hide_timer)
		hide_timer.start()
	elif webcam_frames_received % 30 == 0:  # Update setiap 30 frame
		camera_status_label.text = "Webcam aktif - Frame: " + str(webcam_frames_received)

func _on_webcam_connection_changed(connected: bool):
	"""Callback ketika status koneksi webcam berubah"""
	if connected:
		camera_status_label.text = "✅ Webcam terhubung - Siap deteksi!"
		camera_status_label.modulate = Color(0, 1, 0, 0.9)
		print("Webcam server connected")
	else:
		camera_status_label.text = "❌ Webcam terputus - Cek server Python"
		camera_status_label.modulate = Color(1, 0, 0, 0.9)
		camera_status_label.visible = true
		
		# Jangan gunakan await dalam callback - bisa crash saat node di-destroy
		# Gunakan timer sebagai gantinya
		if webcam_manager and not webcam_manager.get_connection_status():
			var reconnect_timer = Timer.new()
			reconnect_timer.wait_time = 3.0
			reconnect_timer.one_shot = true
			reconnect_timer.timeout.connect(func():
				if is_inside_tree() and webcam_manager and not webcam_manager.get_connection_status():
					camera_status_label.text = "🔄 Mencoba koneksi ulang..."
					camera_status_label.modulate = Color(1, 1, 0, 0.9)
					webcam_manager.connect_to_webcam_server()
				reconnect_timer.queue_free()
			)
			add_child(reconnect_timer)
			reconnect_timer.start()

func _on_webcam_error(message: String):
	"""Callback ketika terjadi error webcam"""
	camera_status_label.text = "❌ Error: " + message
	camera_status_label.modulate = Color(1, 0, 0, 0.9)
	camera_status_label.visible = true
	print("Webcam Error: " + message)

func setup_loading_spinner():
	# Buat spinner loading sederhana
	var spinner_image = Image.create(50, 50, false, Image.FORMAT_RGBA8)
	spinner_image.fill(Color(0, 0, 0, 0))
	
	# Gambar circle dengan gap untuk spinner
	var center = Vector2(25, 25)
	var radius = 20
	
	for angle in range(0, 270, 10):  # 270 derajat untuk gap
		var rad = deg_to_rad(angle)
		var x = int(center.x + cos(rad) * radius)
		var y = int(center.y + sin(rad) * radius)
		
		# Gambar beberapa pixel untuk thickness
		for dx in range(-2, 3):
			for dy in range(-2, 3):
				if x + dx >= 0 and x + dx < 50 and y + dy >= 0 and y + dy < 50:
					spinner_image.set_pixel(x + dx, y + dy, Color(1, 1, 1, 0.8))
	
	var spinner_texture = ImageTexture.new()
	spinner_texture.set_image(spinner_image)
	loading_spinner.texture = spinner_texture

func _process(delta):
	# Animate loading spinner
	if loading_overlay.visible:
		spinner_rotation += delta * 360  # 1 rotation per second
		if spinner_rotation >= 360:
			spinner_rotation -= 360
		loading_spinner.rotation_degrees = spinner_rotation

	# Tampilkan loading overlay dengan animasi
	loading_overlay.visible = true
	spinner_rotation = 0.0
	
	# Gunakan hasil deteksi yang sudah disimpan
	var target_scene = ethnicity_data[detected_ethnicity_result]["scene"]
	
	# Loading animation yang lebih smooth
	var loading_steps = [
		"Memuat aset budaya...",
		"Menyiapkan pengalaman virtual...",
		"Menginisialisasi environment...",
		"Hampir selesai..."
	]
	
	for i in range(loading_steps.size()):
		$LoadingOverlay/LoadingContainer/LoadingLabel.text = loading_steps[i]
		await get_tree().create_timer(0.5).timeout
	
	# Pindah ke scene
	cleanup_resources()
	get_tree().change_scene_to_file(target_scene)

func _on_back_pressed():
	cleanup_resources()
	get_tree().change_scene_to_file("res://Scenes/MainMenu/MainMenu.tscn")

func cleanup_resources():
	"""Bersihkan resources sebelum keluar"""
	print("=== Cleaning up resources ===")

	if webcam_manager:
		print("Disconnecting webcam manager...")
		if webcam_manager.has_method("disconnect_from_server"):
			webcam_manager.disconnect_from_server()
		if is_inside_tree():
			webcam_manager.queue_free()
		webcam_manager = null

	# Hapus baris detection_timer.stop()
	# Hapus baris redirect_timer.stop()

	print("Cleanup complete")

func _notification(what):
	if what == NOTIFICATION_WM_CLOSE_REQUEST or what == NOTIFICATION_PREDELETE:
		cleanup_resources()
