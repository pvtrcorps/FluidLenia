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
@export var show_species: bool = false
@export var show_env: bool = false

# ==============================================================================


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
var tex_advected_species: RID
var tex_advected_aux: RID


# NEW: Species & Expanded Genetics
var tex_species_a: RID
var tex_species_b: RID
var tex_genes_aux_a: RID
var tex_genes_aux_b: RID
var tex_stats: RID # Not used, we use buffer
var buffer_stats: RID
var shader_stats: RID
var pipeline_stats: RID

# Normalization
var shader_sum: RID
var pipeline_sum: RID
var buffer_sum: RID
var total_mass_target: float = 0.0
var global_scale_factor: float = 1.0


# Environment (Temp, Resource, Hazard, _)
var tex_env: RID

signal stats_updated(top_species)




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
var sampler_nearest: RID


# Display (uses ImageTexture for CPU copy from local device)
@onready var display: TextureRect = $TextureRect
var display_image: Image
var display_texture: ImageTexture
var display_waste_image: Image
var display_waste_texture: ImageTexture
var display_species_image: Image
var display_species_texture: ImageTexture
var display_env_image: Image
var display_env_texture: ImageTexture
var display_genes_aux_image: Image
var display_genes_aux_texture: ImageTexture


# ==============================================================================

# ==============================================================================
# LIFECYCLE
# ==============================================================================

func _ready() -> void:
	# Ensure truly random seeds each run
	randomize()
	
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

	# Create Stats Buffer (256 * 24 bytes = 6144 bytes for expanded struct)
	var stats_size = 256 * 24
	var initial_stats = PackedByteArray()
	initial_stats.resize(stats_size)
	initial_stats.fill(0)
	buffer_stats = rd.storage_buffer_create(stats_size, initial_stats)
	
	# Create Sum Buffer (4 bytes uint)
	var sum_init = PackedByteArray()
	sum_init.resize(4)
	sum_init.encode_u32(0, 0)
	buffer_sum = rd.storage_buffer_create(4, sum_init)
	
	# Calculate approximate Target Mass based on density
	# Density 0.35 means 35% of pixels have mass 1.0 (approx)
	# Target = size*size * density
	total_mass_target = float(simulation_size * simulation_size) * initial_density




func _process(_delta: float) -> void:

	if rd == null:
		return
	
	if not paused:
		for i in range(steps_per_frame):
			_simulation_step()
		
		# Forced Normalization: Calculate total mass and update scale factor for NEXT frame
		_dispatch_sum()
		
		# Validar estadísticas periódicamente (e.g. cada 10 frames para no ahogar la CPU)
		if Engine.get_frames_drawn() % 10 == 0:
			_dispatch_stats()


	
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

	var sampler_state_nearest := RDSamplerState.new()
	sampler_state_nearest.min_filter = RenderingDevice.SAMPLER_FILTER_NEAREST
	sampler_state_nearest.mag_filter = RenderingDevice.SAMPLER_FILTER_NEAREST
	sampler_state_nearest.repeat_u = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	sampler_state_nearest.repeat_v = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	sampler_nearest = rd.sampler_create(sampler_state_nearest)


func _create_textures() -> void:
	var fmt_rgba := RDTextureFormat.new()
	fmt_rgba.width = simulation_size
	fmt_rgba.height = simulation_size
	fmt_rgba.format = RenderingDevice.DATA_FORMAT_R32G32B32A32_SFLOAT
	fmt_rgba.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT |
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	)
	
	# 2. Integer Format (for Species IDs) - R32UI
	var fmt_uint := RDTextureFormat.new()
	fmt_uint.width = simulation_size
	fmt_uint.height = simulation_size
	fmt_uint.format = RenderingDevice.DATA_FORMAT_R32_UINT
	fmt_uint.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT |
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	)
	
	tex_living_a = rd.texture_create(fmt_rgba, RDTextureView.new())
	tex_living_b = rd.texture_create(fmt_rgba, RDTextureView.new())
	
	tex_waste_a = rd.texture_create(fmt_rgba, RDTextureView.new())
	tex_waste_b = rd.texture_create(fmt_rgba, RDTextureView.new())
	
	tex_kernel = rd.texture_create(fmt_rgba, RDTextureView.new())

	
	# Genes Aux (Speed, Aggression, etc)
	tex_genes_aux_a = rd.texture_create(fmt_rgba, RDTextureView.new())
	tex_genes_aux_b = rd.texture_create(fmt_rgba, RDTextureView.new())
	
	# Species IDs
	tex_species_a = rd.texture_create(fmt_uint, RDTextureView.new())
	tex_species_b = rd.texture_create(fmt_uint, RDTextureView.new())
	
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
	tex_advected_aux = rd.texture_create(fmt_rgba, RDTextureView.new())
	tex_advected_species = rd.texture_create(fmt_uint, RDTextureView.new())

	# Environment Texture (Static or Dynamic)
	tex_env = rd.texture_create(fmt_rgba, RDTextureView.new())




func _compile_shaders() -> void:
	shader_init = _load_compute_shader("res://shaders/lenia_init.glsl")
	shader_conv = _load_compute_shader("res://shaders/lenia_conv.glsl")
	shader_velocity = _load_compute_shader("res://shaders/lenia_velocity.glsl")
	shader_advect = _load_compute_shader("res://shaders/lenia_advect.glsl")
	shader_growth = _load_compute_shader("res://shaders/lenia_growth.glsl")
	shader_stats = _load_compute_shader("res://shaders/lenia_stats.glsl")
	shader_sum = _load_compute_shader("res://shaders/lenia_sum.glsl")



func _load_compute_shader(path: String) -> RID:
	var shader_file := load(path) as RDShaderFile
	if shader_file == null:
		push_error("Failed to load shader: " + path)
		return RID()
	var spirv := shader_file.get_spirv()
	if spirv == null:
		push_error("Failed to get SPIRV from shader: " + path)
		return RID()
	
	# Check for compilation errors BEFORE trying to create the shader
	var compile_error := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if compile_error != "":
		push_error("SHADER COMPILE ERROR in " + path + ":\n" + compile_error)
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
	if shader_stats.is_valid():
		pipeline_stats = rd.compute_pipeline_create(shader_stats)
	pipeline_sum = rd.compute_pipeline_create(shader_sum)



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
	
	# Create Species display texture (using RF to store the 32-bit ID "as is")
	display_species_image = Image.create(simulation_size, simulation_size, false, Image.FORMAT_RF)
	display_species_texture = ImageTexture.create_from_image(display_species_image)

	# Create Env display texture
	display_env_image = Image.create(simulation_size, simulation_size, false, Image.FORMAT_RGBAF)
	display_env_texture = ImageTexture.create_from_image(display_env_image)

	# Create Aux Genes display texture
	display_genes_aux_image = Image.create(simulation_size, simulation_size, false, Image.FORMAT_RGBAF)
	display_genes_aux_texture = ImageTexture.create_from_image(display_genes_aux_image)


	
	var mat := ShaderMaterial.new()
	mat.shader = load("res://shaders/lenia_render.gdshader")
	mat.set_shader_parameter("u_camera", Vector4(cam_offset.x, cam_offset.y, cam_zoom, 0.0))
	mat.set_shader_parameter("u_show_waste", 1.0 if show_waste else 0.0)
	mat.set_shader_parameter("u_show_species", 1.0 if show_species else 0.0)
	mat.set_shader_parameter("u_show_env", 1.0 if show_env else 0.0)
	mat.set_shader_parameter("u_tex_env", display_env_texture)
	mat.set_shader_parameter("u_tex_genes_aux", display_genes_aux_texture)




	
	display.material = mat
	display.texture = display_texture

func _cleanup_gpu() -> void:
	var rids := [
		tex_living_a, tex_living_b, tex_waste_a, tex_waste_b, 
		tex_kernel, tex_velocity, tex_advected, tex_env,
		shader_init, shader_conv, shader_velocity, shader_advect, shader_growth, shader_stats,

		pipeline_init, pipeline_conv, pipeline_velocity, pipeline_advect, pipeline_growth, pipeline_stats,
		sampler, sampler_nearest, buffer_stats
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
		randf() * 10000.0 + Time.get_ticks_msec() * 0.001, # Better seed variation
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
	
	# NEW BINDINGS FOR SPECIES & GENES
	var u_spec_a := RDUniform.new()
	u_spec_a.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_spec_a.binding = 5
	u_spec_a.add_id(tex_species_a)
	uniforms.append(u_spec_a)
	
	var u_spec_b := RDUniform.new()
	u_spec_b.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_spec_b.binding = 6
	u_spec_b.add_id(tex_species_b)
	uniforms.append(u_spec_b)
	
	var u_genes_a := RDUniform.new()
	u_genes_a.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_genes_a.binding = 7
	u_genes_a.add_id(tex_genes_aux_a)
	uniforms.append(u_genes_a)
	
	var u_genes_b := RDUniform.new()
	u_genes_b.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_genes_b.binding = 8
	u_genes_b.add_id(tex_genes_aux_b)
	uniforms.append(u_genes_b)
	
	# Binding 9: Environment (Write Only)
	var u_env := RDUniform.new()
	u_env.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_env.binding = 9
	u_env.add_id(tex_env)
	uniforms.append(u_env)


	
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
	
	# Read back Environment Texture (Static) once
	if rd.texture_is_valid(tex_env):
		var env_data := rd.texture_get_data(tex_env, 0)
		display_env_image.set_data(simulation_size, simulation_size, false, Image.FORMAT_RGBAF, env_data)
		display_env_texture.update(display_env_image)


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
	
	# NEW: Aux Genes Binding (5)
	var u5 := RDUniform.new()
	u5.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u5.binding = 5
	u5.add_id(sampler)
	if src_living == tex_living_a:
		u5.add_id(tex_genes_aux_a)
	else:
		u5.add_id(tex_genes_aux_b)
	uniforms.append(u5)

	
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
	
	# NEW BINDINGS FOR ADVECTION
	var u4 := RDUniform.new()
	u4.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u4.binding = 4
	u4.add_id(sampler_nearest) # Use Nearest for Integer Texture (Species)
	if src_living == tex_living_a:
		u4.add_id(tex_species_a)
	else:
		u4.add_id(tex_species_b)
	uniforms.append(u4)

	
	var u5 := RDUniform.new()
	u5.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u5.binding = 5
	u5.add_id(sampler)
	if src_living == tex_living_a:
		u5.add_id(tex_genes_aux_a)
	else:
		u5.add_id(tex_genes_aux_b)
	uniforms.append(u5)
	
	var u6 := RDUniform.new()
	u6.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u6.binding = 6
	u6.add_id(tex_advected_species)
	uniforms.append(u6)
	
	var u7 := RDUniform.new()
	u7.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u7.binding = 7
	u7.add_id(tex_advected_aux)
	uniforms.append(u7)

	
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
		global_scale_factor
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
	
	# NEW INPUTS (Advected Winner)
	var u8 := RDUniform.new()
	u8.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u8.binding = 8
	u8.add_id(sampler_nearest) # Use Nearest for Integer (Species)
	u8.add_id(tex_advected_species)
	uniforms.append(u8)

	
	var u9 := RDUniform.new()
	u9.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u9.binding = 9
	u9.add_id(sampler)
	u9.add_id(tex_advected_aux)
	uniforms.append(u9)
	
	# NEW OUTPUTS
	var u10 := RDUniform.new()
	u10.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u10.binding = 10
	if dst_living == tex_living_a:
		u10.add_id(tex_species_a)
	else:
		u10.add_id(tex_species_b)
	uniforms.append(u10)
	
	var u11 := RDUniform.new()
	u11.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u11.binding = 11
	if dst_living == tex_living_a:
		u11.add_id(tex_genes_aux_a)
	else:
		u11.add_id(tex_genes_aux_b)
	uniforms.append(u11)
	
	# Binding 12: Environment (Read Only)
	var u12 := RDUniform.new()
	u12.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u12.binding = 12
	u12.add_id(sampler)
	u12.add_id(tex_env)
	uniforms.append(u12)


	
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

func _dispatch_stats() -> void:
	if not pipeline_stats.is_valid():
		return
		
	# 1. Clear Buffer (Atomic operations aggregate, so we need fresh start)
	var clear_data := PackedByteArray()
	clear_data.resize(256 * 24) # Expanded struct size (6 uints = 24 bytes)
	clear_data.fill(0)
	rd.buffer_update(buffer_stats, 0, clear_data.size(), clear_data)

	
	# 2. Bindings
	var params := PackedFloat32Array([
		float(simulation_size), float(simulation_size),
		0.0, 0.0 # Padding
	])
	var params_buffer := rd.storage_buffer_create(params.size() * 4, params.to_byte_array())
	
	var uniforms: Array[RDUniform] = []
	
	# Binding 0: Species (Read Only) - u_tex_species (r32ui)
	var current_species := tex_species_a if not flip else tex_species_b
	var u0 := RDUniform.new()
	u0.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u0.binding = 0
	u0.add_id(sampler_nearest) # Use Nearest for Integer
	u0.add_id(current_species)
	uniforms.append(u0)

	
	# Binding 1: Genes Aux
	var current_genes := tex_genes_aux_a if not flip else tex_genes_aux_b
	var u1 := RDUniform.new()
	u1.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u1.binding = 1
	u1.add_id(sampler)
	u1.add_id(current_genes)
	uniforms.append(u1)
	
	# Binding 2: StatsBuffer (Storage)
	var u2 := RDUniform.new()
	u2.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u2.binding = 2
	u2.add_id(buffer_stats)
	uniforms.append(u2)
	
	# Binding 3: Params
	var u3 := RDUniform.new()
	u3.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u3.binding = 3
	u3.add_id(params_buffer)
	uniforms.append(u3)
	
	# Binding 4: Living Texture (For Mu)
	var current_living := tex_living_a if not flip else tex_living_b
	var u4 := RDUniform.new()
	u4.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u4.binding = 4
	u4.add_id(sampler)
	u4.add_id(current_living)
	uniforms.append(u4)
	
	var uniform_set := rd.uniform_set_create(uniforms, shader_stats, 0)

	
	# 3. Dispatch
	var groups := ceili(float(simulation_size) / 8.0)
	var compute_list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline_stats)
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	rd.compute_list_dispatch(compute_list, groups, groups, 1)
	rd.compute_list_end()
	
	rd.submit()
	rd.sync() # Wait for GPU to finish so we can read back
	
	rd.free_rid(params_buffer)
	
	# 4. Readback & Parse
	var bytes := rd.buffer_get_data(buffer_stats)
	_parse_and_emit_stats(bytes)

func _dispatch_sum() -> void:
	if not pipeline_sum.is_valid(): return
	
	# 1. Clear Buffer
	var clear_data := PackedByteArray()
	clear_data.resize(4)
	clear_data.encode_u32(0, 0)
	rd.buffer_update(buffer_sum, 0, 4, clear_data)
	
	# 2. Bindings
	# Params (reuse dummy params if strictly needed, or make new)
	var params := PackedFloat32Array([
		float(simulation_size), float(simulation_size),
		0.0, 0.0
	])
	var params_buffer := rd.storage_buffer_create(params.size() * 4, params.to_byte_array())
	
	var uniforms: Array[RDUniform] = []
	
	# Binding 0: Living
	var u0 := RDUniform.new()
	u0.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u0.binding = 0
	u0.add_id(sampler_nearest)
	u0.add_id(tex_living_a if not flip else tex_living_b)
	uniforms.append(u0)
	
	# Binding 1: Waste
	var u1 := RDUniform.new()
	u1.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u1.binding = 1
	u1.add_id(sampler_nearest)
	u1.add_id(tex_waste_a if not flip else tex_waste_b)
	uniforms.append(u1)
	
	# Binding 2: Sum Buffer
	var u2 := RDUniform.new()
	u2.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u2.binding = 2
	u2.add_id(buffer_sum)
	uniforms.append(u2)
	
	# Binding 3: Params
	var u3 := RDUniform.new()
	u3.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u3.binding = 3
	u3.add_id(params_buffer)
	uniforms.append(u3)
	
	var uniform_set := rd.uniform_set_create(uniforms, shader_sum, 0)
	
	# 3. Dispatch
	var groups := ceili(float(simulation_size) / 8.0)
	var compute_list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline_sum)
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	rd.compute_list_dispatch(compute_list, groups, groups, 1)
	rd.compute_list_end()
	
	rd.submit()
	rd.sync() # Wait for readback
	
	rd.free_rid(params_buffer)
	
	# 4. Readback & Calculate Scale Factor
	var bytes := rd.buffer_get_data(buffer_sum)
	var total_mass_fixed = bytes.decode_u32(0)
	var current_total_mass = float(total_mass_fixed) / 1000.0
	
	if current_total_mass > 0.001:
		global_scale_factor = total_mass_target / current_total_mass
		# Soft clamp to avoid explosion
		global_scale_factor = clamp(global_scale_factor, 0.9, 1.1)
	else:
		global_scale_factor = 1.0


func _parse_and_emit_stats(bytes: PackedByteArray) -> void:
	# Struct: uint id, uint count, uint speed_sum, uint aggro_sum, uint mu_sum, uint _pad
	var species_list = []
	
	for i in range(256):
		var offset = i * 24 # Expanded struct: 6 * 4 = 24 bytes
		var s_id = bytes.decode_u32(offset + 0)
		var count = bytes.decode_u32(offset + 4)
		
		if s_id != 0 and count > 1: # Minimum population threshold
			var speed_sum = bytes.decode_u32(offset + 8)
			var aggro_sum = bytes.decode_u32(offset + 12)
			var mu_sum = bytes.decode_u32(offset + 16)
			
			var avg_speed = float(speed_sum) / float(count) / 1000.0
			var avg_aggro = float(aggro_sum) / float(count) / 1000.0
			var avg_mu = float(mu_sum) / float(count) / 1000.0
			
			species_list.append({
				"id": s_id,
				"count": count,
				"avg_speed": avg_speed, 
				"avg_aggro": avg_aggro,
				"avg_mu": avg_mu
			})

	
	# Sort by population (descending)
	species_list.sort_custom(func(a, b): return a.count > b.count)
	
	# Emit Top 5
	var top_5 = species_list.slice(0, 5)
	emit_signal("stats_updated", top_5)

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
	
	# Species Data
	if rd.texture_is_valid(tex_species_a): # Check just in case
		var current_species := tex_species_a if not flip else tex_species_b
		var species_data := rd.texture_get_data(current_species, 0)
		# We interpret the R32_UINT bytes as R32_FLOAT for the Image because Godot doesn't have R32_UINT Image format readily useful for textures
		display_species_image.set_data(simulation_size, simulation_size, false, Image.FORMAT_RF, species_data)
		display_species_texture.update(display_species_image)

	# Aux Genes Data
	if rd.texture_is_valid(tex_genes_aux_a):
		var current_genes := tex_genes_aux_a if not flip else tex_genes_aux_b
		var genes_data := rd.texture_get_data(current_genes, 0)
		display_genes_aux_image.set_data(simulation_size, simulation_size, false, Image.FORMAT_RGBAF, genes_data)
		display_genes_aux_texture.update(display_genes_aux_image)


	
	var mat := display.material as ShaderMaterial
	if mat:
		mat.set_shader_parameter("u_tex_living", display_texture)
		mat.set_shader_parameter("u_tex_waste", display_waste_texture)
		mat.set_shader_parameter("u_tex_species", display_species_texture)
		mat.set_shader_parameter("u_tex_genes_aux", display_genes_aux_texture)
		mat.set_shader_parameter("u_show_waste", 1.0 if show_waste else 0.0)

		mat.set_shader_parameter("u_show_species", 1.0 if show_species else 0.0)
		mat.set_shader_parameter("u_show_env", 1.0 if show_env else 0.0)
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

	# List of specific node names we want to KEEP (Reference only for now)
	# var nodes_to_keep = [
	# 	"ButtonRow/BtnReset", "ButtonRow/BtnClear", "ButtonRow/BtnPause",
	# 	"SliderSPF", "LblSPF",
	# 	"SliderDt", "LblDt",
	# 	"SliderR", "LblR",
	# 	"SliderMutation", "LblMutation", 
	# 	"SliderFriction", "LblFriction",
	# 	"SliderDensity", "LblDensity"
	# ]

	
	# 2. PRO & MINIMAL RESTYLING
	# Create a new container for our modern UI
	var panel_container = PanelContainer.new()
	panel_container.name = "ControlsPanel"
	ui_layer.add_child(panel_container)
	
	# Layout: Left side, fixed width, partial height
	panel_container.set_anchors_preset(Control.PRESET_TOP_LEFT)
	panel_container.position = Vector2(20, 20)
	panel_container.size = Vector2(300, 600)   # Fixed size
	panel_container.modulate.a = 0.95          # Opaque enough to read
	
	# ScrollContainer to handle overflow
	var scroll = ScrollContainer.new()
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	scroll.vertical_scroll_mode = ScrollContainer.SCROLL_MODE_AUTO
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	# CRITICAL: Stop mouse wheel from propagating to camera when hovering/scrolling
	scroll.mouse_filter = Control.MOUSE_FILTER_STOP 
	panel_container.add_child(scroll)

	var vb = VBoxContainer.new()
	vb.add_theme_constant_override("separation", 5) # Compact spacing
	vb.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	vb.custom_minimum_size.x = 260 # Ensure width inside scroll
	
	# A bit of padding inside the scroll
	var margin = MarginContainer.new()
	margin.add_theme_constant_override("margin_left", 10)
	margin.add_theme_constant_override("margin_right", 10)
	margin.add_theme_constant_override("margin_top", 10)
	margin.add_theme_constant_override("margin_bottom", 10)
	margin.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	margin.size_flags_vertical = Control.SIZE_EXPAND_FILL
	
	scroll.add_child(margin)
	margin.add_child(vb)
	
	# --- Header ---
	var header = Label.new()
	header.text = "Emergent Lenia"
	header.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	header.add_theme_font_size_override("font_size", 16)
	header.add_theme_color_override("font_color", Color(0.9, 0.9, 0.9))
	vb.add_child(header)
	vb.add_child(HSeparator.new())
	
	# --- Button Row ---
	var btn_row = HBoxContainer.new()
	btn_row.alignment = BoxContainer.ALIGNMENT_CENTER
	btn_row.add_theme_constant_override("separation", 10)
	vb.add_child(btn_row)
	
	# Reset
	var btn_reset = Button.new()
	btn_reset.text = "RESET"
	btn_reset.custom_minimum_size.x = 60
	btn_reset.pressed.connect(reset)
	btn_row.add_child(btn_reset)
	
	# Pause
	var btn_pause = Button.new()
	btn_pause.text = "PAUSE"
	btn_pause.toggle_mode = true
	btn_pause.button_pressed = paused
	btn_pause.custom_minimum_size.x = 60
	btn_pause.toggled.connect(func(p): toggle_pause())
	btn_row.add_child(btn_pause)
	
	# Clear
	var btn_clear = Button.new()
	btn_clear.text = "CLEAR"
	btn_clear.custom_minimum_size.x = 60
	btn_clear.pressed.connect(clear)
	btn_row.add_child(btn_clear)
	
	vb.add_child(HSeparator.new())

	# Helper to add Slider (Local definition for closure access)
	var add_slider = func(label_text: String, val: float, min_v: float, max_v: float, step: float, callback: Callable):
		var hbox = HBoxContainer.new()
		vb.add_child(hbox)
		
		var lbl = Label.new()
		lbl.text = label_text
		lbl.custom_minimum_size.x = 90
		lbl.modulate = Color(0.7, 0.7, 0.7)
		lbl.add_theme_font_size_override("font_size", 12)
		hbox.add_child(lbl)
		
		var val_lbl = Label.new()
		val_lbl.text = "%.3f" % val
		val_lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
		val_lbl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		val_lbl.add_theme_font_size_override("font_size", 12)
		hbox.add_child(val_lbl)
		
		var slider = HSlider.new()
		slider.min_value = min_v
		slider.max_value = max_v
		slider.step = step
		slider.value = val
		slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		slider.custom_minimum_size.y = 20
		
		# Prevent slider interaction from zooming 
		slider.mouse_filter = Control.MOUSE_FILTER_STOP
		
		slider.value_changed.connect(func(v): 
			callback.call(v)
			val_lbl.text = "%.3f" % v
		)
		vb.add_child(slider)
	
	# --- SIM PARAMETERS ---
	add_slider.call("Sim Speed", float(steps_per_frame), 1.0, 10.0, 1.0, func(v): steps_per_frame = int(v))
	add_slider.call("Delta T", dt, 0.05, 0.5, 0.01, func(v): dt = v)
	
	# --- EVOLUTION ---
	vb.add_child(HSeparator.new())
	var lb_evo = Label.new(); lb_evo.text = "Evolution"; lb_evo.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	lb_evo.add_theme_font_size_override("font_size", 12)
	vb.add_child(lb_evo)
	
	add_slider.call("Mutation", mutation_rate, 0.0, 0.1, 0.001, func(v): mutation_rate = v)
	
	# --- PHYSICS ---
	vb.add_child(HSeparator.new())
	var lb_phys = Label.new(); lb_phys.text = "Physics"; lb_phys.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	lb_phys.add_theme_font_size_override("font_size", 12)
	vb.add_child(lb_phys)

	add_slider.call("Friction", friction, 0.0, 1.0, 0.01, func(v): friction = v)
	add_slider.call("Inertia", inertia, 0.0, 1.0, 0.01, func(v): inertia = v)
	
	# --- ECOSYSTEM ---
	vb.add_child(HSeparator.new())
	var lb_eco = Label.new(); lb_eco.text = "Ecosystem"; lb_eco.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	lb_eco.add_theme_font_size_override("font_size", 12)
	vb.add_child(lb_eco)
	
	add_slider.call("Chemotaxis", chemotaxis, 0.0, 2.0, 0.01, func(v): chemotaxis = v)
	add_slider.call("Eat Rate", eat_rate, 0.0, 1.0, 0.01, func(v): eat_rate = v)
	add_slider.call("Decay", decay, 0.0, 0.1, 0.001, func(v): decay = v)
	add_slider.call("Diet Sel.", diet_selectivity, 0.1, 3.0, 0.1, func(v): diet_selectivity = v)

	# --- INITIALIZATION ---
	vb.add_child(HSeparator.new())
	add_slider.call("Init Dens.", initial_density, 0.05, 0.5, 0.01, func(v): initial_density = v)
	add_slider.call("Init Grid", float(initial_grid), 4.0, 64.0, 1.0, func(v): initial_grid = int(v))

	# --- DISPLAY ---
	vb.add_child(HSeparator.new())
	
	var waste_check = CheckBox.new()
	waste_check.button_pressed = show_waste
	waste_check.text = "Show Waste"
	waste_check.add_theme_font_size_override("font_size", 12)
	waste_check.toggled.connect(func(pressed): show_waste = pressed)
	vb.add_child(waste_check)
	
	var species_check = CheckBox.new()
	species_check.button_pressed = show_species
	species_check.text = "Show Species Colors"
	species_check.add_theme_font_size_override("font_size", 12)
	species_check.toggled.connect(func(pressed): show_species = pressed)
	vb.add_child(species_check)

	var env_check = CheckBox.new()
	env_check.button_pressed = show_env
	env_check.text = "Show Environment"
	env_check.add_theme_font_size_override("font_size", 12)
	env_check.toggled.connect(func(pressed): show_env = pressed)
	vb.add_child(env_check)


	# --- STATS SECTION ---
	vb.add_child(HSeparator.new())
	var lb_stats = Label.new(); lb_stats.text = "Dominant Species"; lb_stats.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	lb_stats.add_theme_font_size_override("font_size", 12)
	vb.add_child(lb_stats)
	
	var stats_container = VBoxContainer.new()
	stats_container.name = "StatsContainer"
	stats_container.add_theme_constant_override("separation", 6)
	vb.add_child(stats_container)
	
	# Pre-allocate 5 rows for stats
	for i in range(5):
		var row = HBoxContainer.new()
		row.name = "Row%d" % i
		row.visible = false # Hidden initially
		
		# ID Color Indicator
		var color_rect = ColorRect.new()
		color_rect.name = "Color"
		color_rect.custom_minimum_size = Vector2(16, 16)
		row.add_child(color_rect)
		
		# Info Label
		var lbl = Label.new()
		lbl.name = "Info"
		lbl.text = "..."
		lbl.add_theme_font_size_override("font_size", 10)
		lbl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		row.add_child(lbl)
		
		stats_container.add_child(row)
	
	# Connect Signal
	if not stats_updated.is_connected(_on_stats_updated):
		stats_updated.connect(_on_stats_updated.bind(stats_container))

	# 4. Hide the old cluttered panel if it exists
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

# Hash function matching shader for UI colors
func _hash_color(id: int) -> Color:
	# Use a simpler hashing strategy consistent with shader
	# Simple LCG-like or param-based hash
	var h = float(id) * 0.123
	var r = fmod(abs(sin(h * 12.9898)), 1.0)
	var g = fmod(abs(sin(h * 78.233)), 1.0)
	var b = fmod(abs(sin(h * 43.758)), 1.0)
	return Color(r, g, b)

func _on_stats_updated(top_species: Array, container: VBoxContainer) -> void:
	if container == null: return
	
	for i in range(5):
		var row = container.get_node_or_null("Row%d" % i)
		if row:
			if i < top_species.size():
				var data = top_species[i]
				row.visible = true
				
				# Update Color - Use avg_mu for Hue to match Phenotype visualization
				var col_rect = row.get_node("Color")
				var hue = data.avg_mu if data.has("avg_mu") else 0.0
				var saturation = 0.3 + data.avg_aggro * 0.7 if data.has("avg_aggro") else 0.8
				var value = clampf(data.avg_speed * 0.6, 0.5, 1.0) if data.has("avg_speed") else 0.9
				col_rect.color = Color.from_hsv(hue, saturation, value)
				
				# Update Text
				var lbl = row.get_node("Info")
				lbl.text = "Pop: %d | Mu: %.2f\nSpd: %.2fx | Agg: %.2f" % [data.count, data.avg_mu if data.has("avg_mu") else 0.0, data.avg_speed, data.avg_aggro]

			else:
				row.visible = false
