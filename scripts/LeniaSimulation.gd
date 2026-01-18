extends Control
class_name LeniaSimulation

## Flow Lenia Simulation Controller
## Manages GPU compute shaders and ping-pong buffers for the Lenia simulation.

# ==============================================================================
# EXPORTED PARAMETERS
# ==============================================================================

@export_group("Simulation")
@export var simulation_size: int = 800
@export var steps_per_frame: int = 3
@export var dt: float = 0.25
@export var kernel_radius: int = 8

@export_group("Evolution")
@export_range(0.0, 0.5) var diet_offset: float = 0.0
@export_range(0.0, 2.0) var chemotaxis: float = 0.8
@export_range(0.0, 0.1) var mutation_rate: float = 0.01
@export_range(0.0, 0.99) var inertia: float = 0.90

@export_group("Physics")
@export_range(0.0, 0.1) var gravity: float = 0.0
@export_range(0.0, 1.0) var friction: float = 0.9
@export_range(0.0, 5.0) var velocity_impact: float = 2.0
@export_range(1.0, 8.0) var immiscibility: float = 4.0
@export var floor_enabled: bool = false

@export_group("Metabolism")
@export_range(0.001, 0.05) var decay: float = 0.012
@export_range(0.0, 1.0) var eat_rate: float = 0.6
@export_range(0.5, 3.0) var diet_selectivity: float = 1.5

@export_group("Initialization")
@export_range(0.05, 0.5) var initial_density: float = 0.35
@export var initial_grid: int = 16

@export_group("Brush")
@export_range(0.0, 1.0) var brush_hue: float = 0.0
@export var brush_size: float = 40.0
@export var brush_mode: int = 1 # 1 = create, -1 = erase

@export_group("Display")
@export var show_waste: bool = true

# ==============================================================================
# INTERNAL STATE
# ==============================================================================

var rd: RenderingDevice
var flip: bool = false
var paused: bool = false
var mouse_painting: bool = false
var mouse_world: Vector2 = Vector2.ZERO

# Camera
var cam_offset: Vector2 = Vector2.ZERO
var cam_zoom: float = 1.0

# GPU Resources
var tex_living_a: RID
var tex_living_b: RID
var tex_waste_a: RID
var tex_waste_b: RID
var tex_kernel: RID

var shader_init: RID
var shader_conv: RID
var shader_flow: RID

var pipeline_init: RID
var pipeline_conv: RID
var pipeline_flow: RID

# Uniform sets (rebuilt each dispatch)
var sampler: RID

# Display
@onready var display: TextureRect = $TextureRect
var display_texture: Texture2DRD
var waste_texture: Texture2DRD

# UI References
@onready var ui_panel: PanelContainer = $UILayer/Panel if has_node("UILayer/Panel") else null

# ==============================================================================
# LIFECYCLE
# ==============================================================================

func _ready() -> void:
	rd = RenderingServer.get_rendering_device()
	if rd == null:
		push_error("RenderingDevice not available. Make sure you're using Forward+ or Mobile renderer.")
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
	
	if not paused or mouse_painting:
		for i in range(steps_per_frame):
			_simulation_step()
	
	_update_display()

func _exit_tree() -> void:
	if rd == null:
		return
	_cleanup_gpu()

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
	var fmt := RDTextureFormat.new()
	fmt.width = simulation_size
	fmt.height = simulation_size
	fmt.format = RenderingDevice.DATA_FORMAT_R32G32B32A32_SFLOAT
	fmt.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT |
		RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	)
	
	tex_living_a = rd.texture_create(fmt, RDTextureView.new())
	tex_living_b = rd.texture_create(fmt, RDTextureView.new())
	tex_waste_a = rd.texture_create(fmt, RDTextureView.new())
	tex_waste_b = rd.texture_create(fmt, RDTextureView.new())
	tex_kernel = rd.texture_create(fmt, RDTextureView.new())

func _compile_shaders() -> void:
	shader_init = _load_compute_shader("res://shaders/lenia_init.glsl")
	shader_conv = _load_compute_shader("res://shaders/lenia_conv.glsl")
	shader_flow = _load_compute_shader("res://shaders/lenia_flow.glsl")

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
	if shader_flow.is_valid():
		pipeline_flow = rd.compute_pipeline_create(shader_flow)

func _setup_display() -> void:
	if display == null:
		push_error("TextureRect not found. Add a TextureRect child named 'TextureRect'.")
		return
	
	display_texture = Texture2DRD.new()
	display_texture.texture_rd_rid = tex_living_a if not flip else tex_living_b
	
	var mat := ShaderMaterial.new()
	mat.shader = load("res://shaders/lenia_render.gdshader")
	mat.set_shader_parameter("u_show_waste", 1.0 if show_waste else 0.0)
	mat.set_shader_parameter("u_camera", Vector4(cam_offset.x, cam_offset.y, cam_zoom, 0.0))
	
	display.material = mat
	display.texture = display_texture

func _cleanup_gpu() -> void:
	var rids := [
		tex_living_a, tex_living_b, tex_waste_a, tex_waste_b, tex_kernel,
		shader_init, shader_conv, shader_flow,
		pipeline_init, pipeline_conv, pipeline_flow,
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
	
	# Create uniform buffer for init params
	var params := PackedFloat32Array([
		float(simulation_size), float(simulation_size), # u_res
		randf() * 100.0, # u_seed
		initial_density, # u_density
		float(initial_grid), # u_initGrid
		0.0, 0.0, 0.0 # padding
	])
	var params_buffer := rd.storage_buffer_create(params.size() * 4, params.to_byte_array())
	
	# Create uniform set
	var uniforms: Array[RDUniform] = []
	
	# Binding 0: out_living
	var u0 := RDUniform.new()
	u0.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u0.binding = 0
	u0.add_id(tex_living_a)
	uniforms.append(u0)
	
	# Binding 1: out_waste
	var u1 := RDUniform.new()
	u1.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u1.binding = 1
	u1.add_id(tex_waste_a)
	uniforms.append(u1)
	
	# Binding 2: params
	var u2 := RDUniform.new()
	u2.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u2.binding = 2
	u2.add_id(params_buffer)
	uniforms.append(u2)
	
	var uniform_set := rd.uniform_set_create(uniforms, shader_init, 0)
	
	# Dispatch
	var groups := ceili(float(simulation_size) / 8.0)
	var compute_list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline_init)
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	rd.compute_list_dispatch(compute_list, groups, groups, 1)
	rd.compute_list_end()
	
	rd.submit()
	rd.sync()
	
	# Cleanup
	rd.free_rid(params_buffer)
	rd.free_rid(uniform_set)
	
	flip = false

func _simulation_step() -> void:
	var src_living := tex_living_a if not flip else tex_living_b
	var src_waste := tex_waste_a if not flip else tex_waste_b
	var dst_living := tex_living_b if not flip else tex_living_a
	var dst_waste := tex_waste_b if not flip else tex_waste_a
	
	# 1. Convolution pass
	_dispatch_convolution(src_living)
	
	# 2. Flow pass
	_dispatch_flow(src_living, src_waste, dst_living, dst_waste)
	
	flip = not flip

func _dispatch_convolution(src_living: RID) -> void:
	if not pipeline_conv.is_valid():
		return
	
	# Params buffer
	var params := PackedFloat32Array([
		float(simulation_size), float(simulation_size), # u_res
		float(kernel_radius), # u_R
		0.0 # padding
	])
	var params_buffer := rd.storage_buffer_create(params.size() * 4, params.to_byte_array())
	
	var uniforms: Array[RDUniform] = []
	
	# Binding 0: u_tex_living (sampler)
	var u0 := RDUniform.new()
	u0.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u0.binding = 0
	u0.add_id(sampler)
	u0.add_id(src_living)
	uniforms.append(u0)
	
	# Binding 1: out_kernel (image)
	var u1 := RDUniform.new()
	u1.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u1.binding = 1
	u1.add_id(tex_kernel)
	uniforms.append(u1)
	
	# Binding 2: params
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
	rd.free_rid(uniform_set)

func _dispatch_flow(src_living: RID, src_waste: RID, dst_living: RID, dst_waste: RID) -> void:
	if not pipeline_flow.is_valid():
		return
	
	# Params buffer - must match shader layout exactly
	var params := PackedFloat32Array([
		float(simulation_size), float(simulation_size), # u_res
		dt, # u_dt
		randf() * 100.0, # u_seed
		decay, # u_decay
		eat_rate, # u_eat_rate
		diet_selectivity, # u_diet_selectivity
		chemotaxis, # u_chemotaxis
		mutation_rate, # u_mutation_rate
		inertia, # u_inertia
		diet_offset, # u_dietOffset
		gravity, # u_gravity
		1.0 if floor_enabled else 0.0, # u_floor
		immiscibility, # u_immiscibility
		friction, # u_friction
		velocity_impact, # u_vel_impact
		0.0, # _pad0
		# Mouse/Brush
		mouse_world.x, mouse_world.y, # u_mouseWorld
		1.0 if mouse_painting else 0.0, # u_mouseClick
		brush_size, # u_brushSize
		brush_hue, # u_brushHue
		float(brush_mode), # u_brushMode
		0.0, 0.0 # _pad1, _pad2
	])
	var params_buffer := rd.storage_buffer_create(params.size() * 4, params.to_byte_array())
	
	var uniforms: Array[RDUniform] = []
	
	# Binding 0: u_tex_living
	var u0 := RDUniform.new()
	u0.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u0.binding = 0
	u0.add_id(sampler)
	u0.add_id(src_living)
	uniforms.append(u0)
	
	# Binding 1: u_tex_waste
	var u1 := RDUniform.new()
	u1.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u1.binding = 1
	u1.add_id(sampler)
	u1.add_id(src_waste)
	uniforms.append(u1)
	
	# Binding 2: u_tex_kernel
	var u2 := RDUniform.new()
	u2.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u2.binding = 2
	u2.add_id(sampler)
	u2.add_id(tex_kernel)
	uniforms.append(u2)
	
	# Binding 3: out_living
	var u3 := RDUniform.new()
	u3.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u3.binding = 3
	u3.add_id(dst_living)
	uniforms.append(u3)
	
	# Binding 4: out_waste
	var u4 := RDUniform.new()
	u4.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u4.binding = 4
	u4.add_id(dst_waste)
	uniforms.append(u4)
	
	# Binding 5: params
	var u5 := RDUniform.new()
	u5.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u5.binding = 5
	u5.add_id(params_buffer)
	uniforms.append(u5)
	
	var uniform_set := rd.uniform_set_create(uniforms, shader_flow, 0)
	
	var groups := ceili(float(simulation_size) / 8.0)
	var compute_list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline_flow)
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	rd.compute_list_dispatch(compute_list, groups, groups, 1)
	rd.compute_list_end()
	
	rd.submit()
	rd.sync()
	
	rd.free_rid(params_buffer)
	rd.free_rid(uniform_set)

func _update_display() -> void:
	if display_texture == null or display == null:
		return
	
	# Update texture RIDs based on current flip state
	var current_living := tex_living_a if not flip else tex_living_b
	var current_waste := tex_waste_a if not flip else tex_waste_b
	display_texture.texture_rd_rid = current_living
	
	# Use persistent waste texture
	if waste_texture == null:
		waste_texture = Texture2DRD.new()
	waste_texture.texture_rd_rid = current_waste
	
	var mat := display.material as ShaderMaterial
	if mat:
		mat.set_shader_parameter("u_tex_living", display_texture)
		mat.set_shader_parameter("u_tex_waste", waste_texture)
		mat.set_shader_parameter("u_show_waste", 1.0 if show_waste else 0.0)
		mat.set_shader_parameter("u_camera", Vector4(cam_offset.x, cam_offset.y, cam_zoom, 0.0))

# ==============================================================================
# INPUT
# ==============================================================================

func _input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		if mb.button_index == MOUSE_BUTTON_LEFT:
			mouse_painting = mb.pressed
		elif mb.button_index == MOUSE_BUTTON_WHEEL_UP:
			cam_zoom = clamp(cam_zoom * 1.1, 0.1, 20.0)
		elif mb.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			cam_zoom = clamp(cam_zoom * 0.9, 0.1, 20.0)
	
	elif event is InputEventMouseMotion:
		var mm := event as InputEventMouseMotion
		_update_mouse_world(mm.position)
		
		if Input.is_mouse_button_pressed(MOUSE_BUTTON_RIGHT):
			cam_offset -= mm.relative / (get_viewport().get_visible_rect().size * cam_zoom)

func _update_mouse_world(screen_pos: Vector2) -> void:
	var viewport_size := get_viewport().get_visible_rect().size
	var mx := screen_pos.x / viewport_size.x
	var my := 1.0 - screen_pos.y / viewport_size.y
	
	# Apply camera transform
	var wx := (mx - 0.5) / cam_zoom + cam_offset.x + 0.5
	var wy := (my - 0.5) / cam_zoom + cam_offset.y + 0.5
	
	mouse_world = Vector2(wx, wy)

# ==============================================================================
# PUBLIC API
# ==============================================================================

func reset() -> void:
	_run_init()

func clear() -> void:
	# Clear all textures to black
	var clear_data := PackedByteArray()
	clear_data.resize(simulation_size * simulation_size * 16) # RGBA32F = 16 bytes per pixel
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

# ==============================================================================
# UI CONNECTION
# ==============================================================================

func _connect_ui() -> void:
	# Get UI nodes - Now inside ScrollContainer
	var vbox = get_node_or_null("UILayer/Panel/Scroll/VBox")
	if vbox == null:
		# Fallback for old scene structure just in case
		vbox = get_node_or_null("UILayer/Panel/VBox")
	
	if vbox == null:
		push_warning("UI VBox not found, skipping UI connection")
		return
	
	# Buttons
	var btn_reset = vbox.get_node_or_null("ButtonRow/BtnReset")
	var btn_clear = vbox.get_node_or_null("ButtonRow/BtnClear")
	var btn_pause = vbox.get_node_or_null("ButtonRow/BtnPause")
	var btn_create = vbox.get_node_or_null("BrushModeRow/BtnCreate")
	var btn_erase = vbox.get_node_or_null("BrushModeRow/BtnErase")
	
	if btn_reset: btn_reset.pressed.connect(reset)
	if btn_clear: btn_clear.pressed.connect(clear)
	if btn_pause: btn_pause.pressed.connect(func(): toggle_pause(); btn_pause.text = "RESUME" if paused else "PAUSE")
	if btn_create: btn_create.pressed.connect(func(): brush_mode = 1)
	if btn_erase: btn_erase.pressed.connect(func(): brush_mode = -1)
	
	# Sliders
	_connect_slider(vbox, "SliderSPF", "LblSPF", "Steps/Frame", func(v): steps_per_frame = int(v), "%d")
	_connect_slider(vbox, "SliderDt", "LblDt", "Delta T", func(v): dt = v, "%.2f")
	_connect_slider(vbox, "SliderR", "LblR", "Kernel Radius", func(v): kernel_radius = int(v), "%d")
	
	_connect_slider(vbox, "SliderDensity", "LblDensity", "Initial Density", func(v): initial_density = v, "%.2f")
	
	_connect_slider(vbox, "SliderDietOffset", "LblDietOffset", "Diet Offset", func(v): diet_offset = v, "%.2f")
	_connect_slider(vbox, "SliderMutation", "LblMutation", "Mutation", func(v): mutation_rate = v, "%.3f")
	_connect_slider(vbox, "SliderInertia", "LblInertia", "Inertia", func(v): inertia = v, "%.2f")
	_connect_slider(vbox, "SliderChemotaxis", "LblChemotaxis", "Chemotaxis", func(v): chemotaxis = v, "%.1f")
	
	_connect_slider(vbox, "SliderDecay", "LblDecay", "Decay", func(v): decay = v, "%.3f")
	_connect_slider(vbox, "SliderEatRate", "LblEatRate", "Eat Rate", func(v): eat_rate = v, "%.2f")
	_connect_slider(vbox, "SliderDietSel", "LblDietSel", "Diet Selectivity", func(v): diet_selectivity = v, "%.1f")
	
	_connect_slider(vbox, "SliderGravity", "LblGravity", "Gravity", func(v): gravity = v, "%.3f")
	_connect_slider(vbox, "SliderFriction", "LblFriction", "Friction", func(v): friction = v, "%.2f")
	_connect_slider(vbox, "SliderVelImpact", "LblVelImpact", "Vel. Impact", func(v): velocity_impact = v, "%.1f")
	_connect_slider(vbox, "SliderImmiscibility", "LblImmiscibility", "Immiscibility", func(v): immiscibility = v, "%.1f")
	
	_connect_slider(vbox, "SliderBrushSize", "LblBrushSize", "Brush Size", func(v): brush_size = v, "%d")
	_connect_slider(vbox, "SliderHue", "LblHue", "Species Hue", func(v): brush_hue = v, "%.2f")
	
	# Checkboxes
	var chk_floor = vbox.get_node_or_null("ChkFloor")
	var chk_waste = vbox.get_node_or_null("ChkWaste")
	
	if chk_floor: chk_floor.toggled.connect(func(pressed): floor_enabled = pressed)
	if chk_waste: chk_waste.toggled.connect(func(pressed): show_waste = pressed)
	
	# Sync initial values
	_sync_sliders_to_values(vbox)

func _connect_slider(vbox: Node, slider_name: String, label_name: String, label_prefix: String, setter: Callable, format: String) -> void:
	var slider = vbox.get_node_or_null(slider_name) as HSlider
	var label = vbox.get_node_or_null(label_name) as Label
	
	if slider and label:
		# Initial label update
		label.text = "%s: %s" % [label_prefix, format % slider.value]
		
		slider.value_changed.connect(func(v):
			setter.call(v)
			label.text = "%s: %s" % [label_prefix, format % v]
		)

func _sync_sliders_to_values(vbox: Node) -> void:
	# Helper to safely set value
	var set_val = func(name, value):
		var s = vbox.get_node_or_null(name) as HSlider
		if s: s.value = value

	set_val.call("SliderSPF", steps_per_frame)
	set_val.call("SliderDt", dt)
	set_val.call("SliderR", kernel_radius)
	set_val.call("SliderDensity", initial_density)
	set_val.call("SliderDietOffset", diet_offset)
	set_val.call("SliderMutation", mutation_rate)
	set_val.call("SliderInertia", inertia)
	set_val.call("SliderChemotaxis", chemotaxis)
	set_val.call("SliderDecay", decay)
	set_val.call("SliderEatRate", eat_rate)
	set_val.call("SliderDietSel", diet_selectivity)
	set_val.call("SliderGravity", gravity)
	set_val.call("SliderFriction", friction)
	set_val.call("SliderVelImpact", velocity_impact)
	set_val.call("SliderImmiscibility", immiscibility)
	set_val.call("SliderBrushSize", brush_size)
	set_val.call("SliderHue", brush_hue)
