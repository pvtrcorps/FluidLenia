extends Control
class_name LeniaSimulation

## Flow Lenia Simulation Controller (Mass Conserving)
## Uses a LOCAL RenderingDevice for compute, with texture copy to global for display.

# ==============================================================================
# EXPORTED PARAMETERS
# ==============================================================================

@export_group("Simulation")
@export var simulation_size: int = 512
@export var steps_per_frame: int = 3
@export var dt: float = 0.25
@export var kernel_radius: int = 8

@export_group("Flow")
@export_range(0.0, 1.0) var friction: float = 0.8

@export_group("Evolution")
@export_range(0.0, 0.1) var mutation_rate: float = 0.01

@export_group("Initialization")
@export_range(0.05, 0.5) var initial_density: float = 0.35
@export var initial_grid: int = 16

@export_group("Ecosystem")
@export_range(0.0, 2.0) var chemotaxis: float = 0.8
@export_range(0.0, 1.0) var eat_rate: float = 0.6
@export_range(0.0, 1.0) var decay: float = 0.012
@export_range(0.0, 1.0) var inertia: float = 0.9
@export_range(0.1, 3.0) var diet_selectivity: float = 1.5

@export_group("Display")
@export var show_waste: bool = true

# ==============================================================================
# INTERNAL STATE
# ==============================================================================

var rd: RenderingDevice  # LOCAL device for compute
var flip: bool = false
var paused: bool = false

# Camera
var cam_offset: Vector2 = Vector2.ZERO
var cam_zoom: float = 1.0

# GPU Resources - Textures (on LOCAL device)
var tex_living_a: RID
var tex_living_b: RID
var tex_waste_a: RID
var tex_waste_b: RID
var tex_kernel: RID
var tex_velocity: RID
var tex_advected: RID

# Shaders
var shader_init: RID
var shader_conv: RID
var shader_velocity: RID
var shader_advect: RID
var shader_growth: RID

# Pipelines
var pipeline_init: RID
var pipeline_conv: RID
var pipeline_velocity: RID
var pipeline_advect: RID
var pipeline_growth: RID

# Sampler
var sampler: RID

# Display (uses ImageTexture for CPU copy from local device)
@onready var display: TextureRect = $TextureRect
var display_image: Image
var display_texture: ImageTexture
var display_waste_image: Image
var display_waste_texture: ImageTexture

# ==============================================================================
# LIFECYCLE
# ==============================================================================

func _ready() -> void:
	# Create LOCAL rendering device (allows submit/sync)
	rd = RenderingServer.create_local_rendering_device()
	if rd == null:
		push_error("Could not create local RenderingDevice.")
		return
	
	_create_sampler()
	_create_textures()
	_compile_shaders()
	_create_pipelines()
	_setup_display()
	_connect_ui()
	_run_init()

func _process(_delta: float) -> void:
	if rd == null:
		return
	
	if not paused:
		for i in range(steps_per_frame):
			_simulation_step()
	
	_update_display()

func _exit_tree() -> void:
	if rd == null:
		return
	_cleanup_gpu()
	rd.free()

# ==============================================================================
# GPU SETUP
# ==============================================================================

func _create_sampler() -> void:
	var sampler_state := RDSamplerState.new()
	sampler_state.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	sampler_state.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	sampler_state.repeat_u = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	sampler_state.repeat_v = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	sampler = rd.sampler_create(sampler_state)

func _create_textures() -> void:
	var fmt_rgba := RDTextureFormat.new()
	fmt_rgba.width = simulation_size
	fmt_rgba.height = simulation_size
	fmt_rgba.format = RenderingDevice.DATA_FORMAT_R32G32B32A32_SFLOAT
	fmt_rgba.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT |
		RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	)
	
	tex_living_a = rd.texture_create(fmt_rgba, RDTextureView.new())
	tex_living_b = rd.texture_create(fmt_rgba, RDTextureView.new())
	tex_waste_a = rd.texture_create(fmt_rgba, RDTextureView.new())
	tex_waste_b = rd.texture_create(fmt_rgba, RDTextureView.new())
	tex_kernel = rd.texture_create(fmt_rgba, RDTextureView.new())
	tex_velocity = rd.texture_create(fmt_rgba, RDTextureView.new())
	
	var fmt_int := RDTextureFormat.new()
	fmt_int.width = simulation_size
	fmt_int.height = simulation_size
	fmt_int.format = RenderingDevice.DATA_FORMAT_R32_SINT
	fmt_int.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT
	)
	
	tex_advected = rd.texture_create(fmt_rgba, RDTextureView.new())

func _compile_shaders() -> void:
	shader_init = _load_compute_shader("res://shaders/lenia_init.glsl")
	shader_conv = _load_compute_shader("res://shaders/lenia_conv.glsl")
	shader_velocity = _load_compute_shader("res://shaders/lenia_velocity.glsl")
	shader_advect = _load_compute_shader("res://shaders/lenia_advect.glsl")
	shader_growth = _load_compute_shader("res://shaders/lenia_growth.glsl")

func _load_compute_shader(path: String) -> RID:
	var shader_file := load(path) as RDShaderFile
	if shader_file == null:
		push_error("Failed to load shader: " + path)
		return RID()
	var spirv := shader_file.get_spirv()
	if spirv == null:
		push_error("Failed to get SPIRV from shader: " + path)
		return RID()
	return rd.shader_create_from_spirv(spirv)

func _create_pipelines() -> void:
	if shader_init.is_valid():
		pipeline_init = rd.compute_pipeline_create(shader_init)
	if shader_conv.is_valid():
		pipeline_conv = rd.compute_pipeline_create(shader_conv)
	if shader_velocity.is_valid():
		pipeline_velocity = rd.compute_pipeline_create(shader_velocity)
	if shader_advect.is_valid():
		pipeline_advect = rd.compute_pipeline_create(shader_advect)
	if shader_growth.is_valid():
		pipeline_growth = rd.compute_pipeline_create(shader_growth)

func _setup_display() -> void:
	if display == null:
		push_error("TextureRect not found.")
		return
	
	# Create CPU-side image and texture for display
	display_image = Image.create(simulation_size, simulation_size, false, Image.FORMAT_RGBAF)
	display_texture = ImageTexture.create_from_image(display_image)
	
	# Create waste display texture
	display_waste_image = Image.create(simulation_size, simulation_size, false, Image.FORMAT_RGBAF)
	display_waste_texture = ImageTexture.create_from_image(display_waste_image)
	
	var mat := ShaderMaterial.new()
	mat.shader = load("res://shaders/lenia_render.gdshader")
	mat.set_shader_parameter("u_camera", Vector4(cam_offset.x, cam_offset.y, cam_zoom, 0.0))
	mat.set_shader_parameter("u_show_waste", 1.0 if show_waste else 0.0)
	
	display.material = mat
	display.texture = display_texture

func _cleanup_gpu() -> void:
	var rids := [
		tex_living_a, tex_living_b, tex_waste_a, tex_waste_b, 
		tex_kernel, tex_velocity, tex_advected,
		shader_init, shader_conv, shader_velocity, shader_advect, shader_growth,
		pipeline_init, pipeline_conv, pipeline_velocity, pipeline_advect, pipeline_growth,
		sampler
	]
	for r in rids:
		if r.is_valid():
			rd.free_rid(r)

# ==============================================================================
# SIMULATION
# ==============================================================================

func _run_init() -> void:
	if not pipeline_init.is_valid():
		push_error("Init pipeline not valid")
		return
	
	var params := PackedFloat32Array([
		float(simulation_size), float(simulation_size),
		randf() * 100.0,
		initial_density,
		float(initial_grid),
		0.0, 0.0, 0.0
	])
	var params_buffer := rd.storage_buffer_create(params.size() * 4, params.to_byte_array())
	
	var uniforms: Array[RDUniform] = []
	
	var u0 := RDUniform.new()
	u0.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u0.binding = 0
	u0.add_id(tex_living_a)
	uniforms.append(u0)
	
	var u1 := RDUniform.new()
	u1.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u1.binding = 1
	u1.add_id(tex_living_b)
	uniforms.append(u1)
	
	var u_wa := RDUniform.new()
	u_wa.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_wa.binding = 2
	u_wa.add_id(tex_waste_a)
	uniforms.append(u_wa)
	
	var u_wb := RDUniform.new()
	u_wb.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_wb.binding = 3
	u_wb.add_id(tex_waste_b)
	uniforms.append(u_wb)
	
	var u2 := RDUniform.new()
	u2.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u2.binding = 4
	u2.add_id(params_buffer)
	uniforms.append(u2)
	
	var uniform_set := rd.uniform_set_create(uniforms, shader_init, 0)
	
	var groups := ceili(float(simulation_size) / 8.0)
	var compute_list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline_init)
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	rd.compute_list_dispatch(compute_list, groups, groups, 1)
	rd.compute_list_end()
	
	rd.submit()
	rd.sync()
	
	rd.free_rid(params_buffer)
	
	flip = false

func _simulation_step() -> void:
	var src_living := tex_living_a if not flip else tex_living_b
	var dst_living := tex_living_b if not flip else tex_living_a
	
	_clear_accumulator()
	_dispatch_convolution(src_living)
	
	var src_waste := tex_waste_a if not flip else tex_waste_b
	var dst_waste := tex_waste_b if not flip else tex_waste_a
	
	_dispatch_velocity(src_living, src_waste)
	_dispatch_advect(src_living)
	_dispatch_growth(src_living, dst_living, src_waste, dst_waste)
	
	flip = not flip

func _clear_accumulator() -> void:
	# No strictly needed if we overwrite in Advect, but good practice if logic changes
	pass

func _dispatch_convolution(src_living: RID) -> void:
	if not pipeline_conv.is_valid():
		return
	
	var params := PackedFloat32Array([
		float(simulation_size), float(simulation_size),
		float(kernel_radius),
		0.0
	])
	var params_buffer := rd.storage_buffer_create(params.size() * 4, params.to_byte_array())
	
	var uniforms: Array[RDUniform] = []
	
	var u0 := RDUniform.new()
	u0.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u0.binding = 0
	u0.add_id(sampler)
	u0.add_id(src_living)
	uniforms.append(u0)
	
	var u1 := RDUniform.new()
	u1.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u1.binding = 1
	u1.add_id(tex_kernel)
	uniforms.append(u1)
	
	var u2 := RDUniform.new()
	u2.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u2.binding = 2
	u2.add_id(params_buffer)
	uniforms.append(u2)
	
	var uniform_set := rd.uniform_set_create(uniforms, shader_conv, 0)
	
	var groups := ceili(float(simulation_size) / 8.0)
	var compute_list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline_conv)
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	rd.compute_list_dispatch(compute_list, groups, groups, 1)
	rd.compute_list_end()
	
	rd.submit()
	rd.sync()
	
	rd.free_rid(params_buffer)

func _dispatch_velocity(src_living: RID, src_waste: RID) -> void:
	if not pipeline_velocity.is_valid():
		return
	
	var params := PackedFloat32Array([
		float(simulation_size), float(simulation_size),
		dt,
		friction,
		chemotaxis, 
		0.0, 0.0, 0.0 # Padding
	])
	var params_buffer := rd.storage_buffer_create(params.size() * 4, params.to_byte_array())
	
	var uniforms: Array[RDUniform] = []
	
	var u0 := RDUniform.new()
	u0.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u0.binding = 0
	u0.add_id(sampler)
	u0.add_id(tex_kernel)
	uniforms.append(u0)
	
	var u1 := RDUniform.new()
	u1.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u1.binding = 1
	u1.add_id(sampler)
	u1.add_id(src_living)
	uniforms.append(u1)
	
	var u2 := RDUniform.new()
	u2.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u2.binding = 2
	u2.add_id(tex_velocity)
	uniforms.append(u2)
	
	var u3 := RDUniform.new()
	u3.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u3.binding = 3
	u3.add_id(sampler)
	u3.add_id(src_waste) # Requires updating _dispatch_velocity signature!
	uniforms.append(u3)
	
	var u4 := RDUniform.new()
	u4.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u4.binding = 4
	u4.add_id(params_buffer)
	uniforms.append(u4)
	
	var uniform_set := rd.uniform_set_create(uniforms, shader_velocity, 0)
	
	var groups := ceili(float(simulation_size) / 8.0)
	var compute_list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline_velocity)
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	rd.compute_list_dispatch(compute_list, groups, groups, 1)
	rd.compute_list_end()
	
	rd.submit()
	rd.sync()
	
	rd.free_rid(params_buffer)

func _dispatch_advect(src_living: RID) -> void:
	if not pipeline_advect.is_valid():
		return
	
	var params := PackedFloat32Array([
		float(simulation_size), float(simulation_size),
		dt,
		0.0
	])
	var params_buffer := rd.storage_buffer_create(params.size() * 4, params.to_byte_array())
	
	var uniforms: Array[RDUniform] = []
	
	var u0 := RDUniform.new()
	u0.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u0.binding = 0
	u0.add_id(sampler)
	u0.add_id(src_living)
	uniforms.append(u0)
	
	var u1 := RDUniform.new()
	u1.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u1.binding = 1
	u1.add_id(sampler)
	u1.add_id(tex_velocity)
	uniforms.append(u1)
	
	var u2 := RDUniform.new()
	u2.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u2.binding = 2
	u2.add_id(tex_advected)
	uniforms.append(u2)
	
	var u3 := RDUniform.new()
	u3.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u3.binding = 3
	u3.add_id(params_buffer)
	uniforms.append(u3)
	
	var uniform_set := rd.uniform_set_create(uniforms, shader_advect, 0)
	
	var groups := ceili(float(simulation_size) / 8.0)
	var compute_list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline_advect)
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	rd.compute_list_dispatch(compute_list, groups, groups, 1)
	rd.compute_list_end()
	
	rd.submit()
	rd.sync()
	
	rd.free_rid(params_buffer)

func _dispatch_growth(src_living: RID, dst_living: RID, src_waste: RID, dst_waste: RID) -> void:
	if not pipeline_growth.is_valid():
		return
	
	# Removed brush params from buffer pack
	var params := PackedFloat32Array([
		float(simulation_size), float(simulation_size),
		dt,
		randf() * 100.0,
		mutation_rate,
		eat_rate,
		decay,
		inertia,
		diet_selectivity,
		0.0, 0.0, 0.0 # Padding to 16 bytes
	])
	var params_buffer := rd.storage_buffer_create(params.size() * 4, params.to_byte_array())
	
	var uniforms: Array[RDUniform] = []
	
	# Binding 0: Advected
	var u0 := RDUniform.new()
	u0.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u0.binding = 0
	u0.add_id(sampler)
	u0.add_id(tex_advected)
	uniforms.append(u0)
	
	# Binding 1: Living Prev
	var u1 := RDUniform.new()
	u1.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u1.binding = 1
	u1.add_id(sampler)
	u1.add_id(src_living)
	uniforms.append(u1)
	
	# Binding 2: Out Living
	var u2 := RDUniform.new()
	u2.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u2.binding = 2
	u2.add_id(dst_living)
	uniforms.append(u2)
	
	# Binding 3: Kernel
	var u3 := RDUniform.new()
	u3.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u3.binding = 3
	u3.add_id(sampler)
	u3.add_id(tex_kernel)
	uniforms.append(u3)
	
	# Binding 5: Src Waste
	var u5 := RDUniform.new()
	u5.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u5.binding = 5
	u5.add_id(sampler)
	u5.add_id(src_waste) 
	uniforms.append(u5)
	
	# Binding 6: Dst Waste
	var u6 := RDUniform.new()
	u6.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u6.binding = 6
	u6.add_id(dst_waste) 
	uniforms.append(u6)
	
	# Binding 7: Params
	var u7 := RDUniform.new()
	u7.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u7.binding = 7
	u7.add_id(params_buffer)
	uniforms.append(u7)
	
	var uniform_set := rd.uniform_set_create(uniforms, shader_growth, 0)
	
	var groups := ceili(float(simulation_size) / 8.0)
	var compute_list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline_growth)
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	rd.compute_list_dispatch(compute_list, groups, groups, 1)
	rd.compute_list_end()
	
	rd.submit()
	rd.sync()
	
	rd.free_rid(params_buffer)

func _update_display() -> void:
	if display_texture == null or display == null:
		return
	
	# Read texture data from LOCAL device
	var current_living := tex_living_a if not flip else tex_living_b
	var current_waste := tex_waste_a if not flip else tex_waste_b
	
	var tex_data := rd.texture_get_data(current_living, 0)
	var waste_data := rd.texture_get_data(current_waste, 0)
	
	# Convert to Image (RGBAF format = 16 bytes per pixel)
	display_image.set_data(simulation_size, simulation_size, false, Image.FORMAT_RGBAF, tex_data)
	display_texture.update(display_image)
	
	display_waste_image.set_data(simulation_size, simulation_size, false, Image.FORMAT_RGBAF, waste_data)
	display_waste_texture.update(display_waste_image)
	
	var mat := display.material as ShaderMaterial
	if mat:
		mat.set_shader_parameter("u_tex_living", display_texture)
		mat.set_shader_parameter("u_tex_waste", display_waste_texture)
		mat.set_shader_parameter("u_show_waste", 1.0 if show_waste else 0.0)
		mat.set_shader_parameter("u_camera", Vector4(cam_offset.x, cam_offset.y, cam_zoom, 0.0))

# ==============================================================================
# INPUT
# ==============================================================================

func _input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		if mb.button_index == MOUSE_BUTTON_WHEEL_UP:
			cam_zoom = clamp(cam_zoom * 1.1, 0.1, 20.0)
		elif mb.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			cam_zoom = clamp(cam_zoom * 0.9, 0.1, 20.0)
	
	elif event is InputEventMouseMotion:
		var mm := event as InputEventMouseMotion
		
		if Input.is_mouse_button_pressed(MOUSE_BUTTON_RIGHT):
			cam_offset -= mm.relative / (get_viewport().get_visible_rect().size * cam_zoom)

# ==============================================================================
# PUBLIC API
# ==============================================================================

func reset() -> void:
	_run_init()

func clear() -> void:
	var clear_data := PackedByteArray()
	clear_data.resize(simulation_size * simulation_size * 16)
	clear_data.fill(0)
	
	rd.texture_update(tex_living_a, 0, clear_data)
	rd.texture_update(tex_living_b, 0, clear_data)
	rd.texture_update(tex_waste_a, 0, clear_data)
	rd.texture_update(tex_waste_b, 0, clear_data)

func toggle_pause() -> void:
	paused = not paused

# ==============================================================================
# UI CONNECTION
# ==============================================================================

func _connect_ui() -> void:
	var ui_layer = get_node_or_null("UILayer")
	if ui_layer == null:
		push_warning("UILayer not found")
		return

	# 1. Identify valid controls (Buttons & Sliders)
	# We assume the old structure is UILayer/Panel/...
	var old_panel = ui_layer.get_node_or_null("Panel")
	var old_vbox = null
	if old_panel:
		# Try to find the VBox where controls live
		old_vbox = old_panel.get_node_or_null("Scroll/VBox")
		if old_vbox == null:
			old_vbox = old_panel.get_node_or_null("VBox")
	
	if old_vbox == null:
		push_warning("Could not find old VBox to retrieve controls from.")
		return

	# List of specific node names we want to KEEP
	var nodes_to_keep = [
		"ButtonRow/BtnReset", "ButtonRow/BtnClear", "ButtonRow/BtnPause",
		"SliderSPF", "LblSPF",
		"SliderDt", "LblDt",
		"SliderR", "LblR",
		"SliderMutation", "LblMutation", 
		"SliderFriction", "LblFriction",
		"SliderDensity", "LblDensity"
	]
	
	# 2. PRO & MINIMAL RESTYLING
	# Create a new container for our modern UI
	var main_panel = PanelContainer.new()
	main_panel.name = "ModernPanel"
	
	# Style: Floating Card, Dark Glass / Material Look
	var style = StyleBoxFlat.new()
	style.bg_color = Color(0.12, 0.12, 0.12, 0.95) # Deep matte grey
	style.corner_radius_top_left = 12
	style.corner_radius_top_right = 12
	style.corner_radius_bottom_right = 12
	style.corner_radius_bottom_left = 12
	style.shadow_size = 8
	style.shadow_color = Color(0, 0, 0, 0.4)
	style.content_margin_left = 20
	style.content_margin_right = 20
	style.content_margin_top = 20
	style.content_margin_bottom = 20
	main_panel.add_theme_stylebox_override("panel", style)
	
	# Layout: Absolute positioning (Top Left)
	main_panel.set_anchors_and_offsets_preset(Control.PRESET_TOP_LEFT)
	main_panel.position = Vector2(24, 24)
	main_panel.custom_minimum_size = Vector2(280, 0) # Fixed width for clean look
	
	ui_layer.add_child(main_panel)
	
	# Content Layout
	var content = VBoxContainer.new()
	content.add_theme_constant_override("separation", 16) # Generous spacing
	main_panel.add_child(content)
	
	# Title
	var title = Label.new()
	title.text = "Flow Lenia"
	title.add_theme_font_size_override("font_size", 18)
	title.add_theme_color_override("font_color", Color(0.9, 0.9, 0.9))
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	content.add_child(title)
	
	# Separator
	var sep = HSeparator.new()
	sep.modulate = Color(1, 1, 1, 0.2)
	content.add_child(sep)
	
	# 3. Harvest and Reparent Controls
	
	# Section: Actions
	var action_box = HBoxContainer.new()
	action_box.alignment = BoxContainer.ALIGNMENT_CENTER
	action_box.add_theme_constant_override("separation", 10)
	content.add_child(action_box)
	
	var row_nodes = old_vbox.get_node_or_null("ButtonRow")
	if row_nodes:
		var btn_reset = row_nodes.get_node_or_null("BtnReset")
		var btn_clear = row_nodes.get_node_or_null("BtnClear") 
		var btn_pause = row_nodes.get_node_or_null("BtnPause")
		
		# Move them to new box
		if btn_reset: 
			btn_reset.get_parent().remove_child(btn_reset)
			action_box.add_child(btn_reset)
			_style_button(btn_reset, "RESET")
			btn_reset.pressed.connect(reset)
			
		if btn_pause:
			btn_pause.get_parent().remove_child(btn_pause)
			action_box.add_child(btn_pause)
			_style_button(btn_pause, "PAUSE")
			btn_pause.pressed.connect(func(): toggle_pause(); btn_pause.text = "RESUME" if paused else "PAUSE")
			
		if btn_clear:
			btn_clear.get_parent().remove_child(btn_clear)
			action_box.add_child(btn_clear)
			_style_button(btn_clear, "CLEAR")
			btn_clear.pressed.connect(clear)
	
	# Section: Parameters
	var param_box = VBoxContainer.new()
	param_box.add_theme_constant_override("separation", 12)
	content.add_child(param_box)
	
	_reparent_slider_group(old_vbox, param_box, "SliderSPF", "LblSPF", "Sim Speed", 
		"Pasos de simulación por frame. Más alto = más rápido.",
		steps_per_frame, func(v): steps_per_frame = int(v), "%dx")
		
	_reparent_slider_group(old_vbox, param_box, "SliderDt", "LblDt", "Delta T", 
		"Velocidad de integración del tiempo. Afecta estabilidad.",
		dt, func(v): dt = v, "%.2f")
		
	_reparent_slider_group(old_vbox, param_box, "SliderR", "LblR", "Kernel Radius", 
		"Radio de percepción de las células en píxeles.",
		kernel_radius, func(v): kernel_radius = int(v), "%d px")
		
	_reparent_slider_group(old_vbox, param_box, "SliderMutation", "LblMutation", "Mutation Rate", 
		"Probabilidad de cambio aleatorio en la genética.",
		mutation_rate, func(v): mutation_rate = v, "%.3f")
		
	_reparent_slider_group(old_vbox, param_box, "SliderFriction", "LblFriction", "Flow Friction", 
		"Resistencia del fluido al movimiento (0=libre, 1=viscoso).",
		friction, func(v): friction = v, "%.2f")
		
	_reparent_slider_group(old_vbox, param_box, "SliderDensity", "LblDensity", "Init Density", 
		"Probabilidad de que un bloque inicie ocupado.",
		initial_density, func(v): initial_density = v, "%.2f")

	# New Params
	_create_slider(param_box, "Init Grid", "Tamaño de la cuadrícula de inicialización.", 1.0, 64.0, 1.0, initial_grid, func(v): initial_grid = int(v))
	_create_slider(param_box, "Chemotaxis", "Atracción hacia residuos (comida).", 0.0, 2.0, 0.1, chemotaxis, func(v): chemotaxis = v)
	_create_slider(param_box, "Eat Rate", "Velocidad de consumo de residuos.", 0.0, 1.0, 0.05, eat_rate, func(v): eat_rate = v)
	_create_slider(param_box, "Decay", "Mortalidad natural de las células.", 0.001, 0.1, 0.001, decay, func(v): decay = v)
	_create_slider(param_box, "Inertia", "Resistencia a cambios de dirección.", 0.0, 1.0, 0.05, inertia, func(v): inertia = v)
	_create_slider(param_box, "Diet Selectivity", "Estrictez de la dieta.", 0.1, 3.0, 0.1, diet_selectivity, func(v): diet_selectivity = v)
	
	# Waste toggle checkbox
	var waste_check_container = HBoxContainer.new()
	param_box.add_child(waste_check_container)
	
	var waste_check = CheckBox.new()
	waste_check.button_pressed = show_waste
	waste_check.text = "Ver Residuos (Comida)"
	waste_check.tooltip_text = "Alternar visualización de la capa de residuos."
	waste_check.toggled.connect(func(pressed): show_waste = pressed)
	waste_check_container.add_child(waste_check)

	# 4. Hide the old cluttered panel
	if old_panel:
		old_panel.visible = false

func _style_button(btn: Button, label: String) -> void:
	btn.text = label
	btn.custom_minimum_size = Vector2(70, 32)
	# Remove default icons if any to keep it clean
	btn.icon = null 
	btn.flat = false

func _reparent_slider_group(source_parent: Node, target_parent: Node, slider_name: String, label_name: String, title: String, tooltip: String, current_val: Variant, setter: Callable, format: String) -> void:
	var slider = source_parent.get_node_or_null(slider_name) as HSlider
	var original_label = source_parent.get_node_or_null(label_name) as Label
	
	if slider:
		# Create a clean container for this parameter
		var container = VBoxContainer.new()
		container.add_theme_constant_override("separation", 4)
		target_parent.add_child(container)
		
		# Header: Title + Value
		var header = HBoxContainer.new()
		container.add_child(header)
		
		var title_lbl = Label.new()
		title_lbl.text = title
		title_lbl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		title_lbl.add_theme_color_override("font_color", Color(0.7, 0.7, 0.7))
		title_lbl.add_theme_font_size_override("font_size", 12) # Small uppercase-like label
		header.add_child(title_lbl)
		
		var value_lbl = Label.new()
		value_lbl.text = format % slider.value
		value_lbl.add_theme_font_size_override("font_size", 12)
		value_lbl.add_theme_color_override("font_color", Color(1, 1, 1))
		header.add_child(value_lbl)
		
		# Update value to match script export (code authority)
		slider.value = float(current_val)
		
		# Tooltip
		slider.tooltip_text = tooltip
		container.tooltip_text = tooltip
		
		# Move Slider
		slider.get_parent().remove_child(slider)
		container.add_child(slider)
		slider.size_flags_horizontal = Control.SIZE_FILL
		
		# Update Label immediately
		value_lbl.text = format % slider.value
		
		# Logic
		slider.value_changed.connect(func(v):
			setter.call(v)
			value_lbl.text = format % v
		)
		
		# Hide original label if it existed separately and wasn't reparented (we made a new one)
		if original_label:
			original_label.visible = false

func _create_slider(parent: Node, title: String, tooltip: String, min_val: float, max_val: float, step: float, default: float, callback: Callable) -> void:
	var container = VBoxContainer.new()
	container.add_theme_constant_override("separation", 4)
	container.tooltip_text = tooltip
	parent.add_child(container)
	
	var header = HBoxContainer.new()
	container.add_child(header)
	
	var title_lbl = Label.new()
	title_lbl.text = title
	title_lbl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	title_lbl.add_theme_color_override("font_color", Color(0.7, 0.7, 0.7))
	title_lbl.add_theme_font_size_override("font_size", 12)
	header.add_child(title_lbl)
	
	var value_lbl = Label.new()
	value_lbl.text = "%.3f" % default
	value_lbl.add_theme_font_size_override("font_size", 12)
	header.add_child(value_lbl)
	
	var slider = HSlider.new()
	slider.min_value = min_val
	slider.max_value = max_val
	slider.step = step
	slider.value = default
	container.add_child(slider)
	
	slider.value_changed.connect(func(v):
		callback.call(v)
		value_lbl.text = "%.3f" % v
	)
