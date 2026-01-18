extends Control
## UI Controller for Flow Lenia
## Connects UI elements to the LeniaSimulation parameters

@onready var simulation: LeniaSimulation = get_parent()

# Sliders
@onready var slider_dt: HSlider = $"UILayer/Panel/VBox/SliderDt"
@onready var slider_r: HSlider = $"UILayer/Panel/VBox/SliderR"
@onready var slider_mutation: HSlider = $"UILayer/Panel/VBox/SliderMutation"
@onready var slider_inertia: HSlider = $"UILayer/Panel/VBox/SliderInertia"
@onready var slider_chemotaxis: HSlider = $"UILayer/Panel/VBox/SliderChemotaxis"
@onready var slider_gravity: HSlider = $"UILayer/Panel/VBox/SliderGravity"
@onready var slider_friction: HSlider = $"UILayer/Panel/VBox/SliderFriction"
@onready var slider_hue: HSlider = $"UILayer/Panel/VBox/SliderHue"

# Labels
@onready var lbl_dt: Label = $"UILayer/Panel/VBox/LblDt"
@onready var lbl_r: Label = $"UILayer/Panel/VBox/LblR"
@onready var lbl_mutation: Label = $"UILayer/Panel/VBox/LblMutation"
@onready var lbl_inertia: Label = $"UILayer/Panel/VBox/LblInertia"
@onready var lbl_chemotaxis: Label = $"UILayer/Panel/VBox/LblChemotaxis"
@onready var lbl_gravity: Label = $"UILayer/Panel/VBox/LblGravity"
@onready var lbl_friction: Label = $"UILayer/Panel/VBox/LblFriction"
@onready var lbl_hue: Label = $"UILayer/Panel/VBox/LblHue"

# Buttons
@onready var btn_reset: Button = $"UILayer/Panel/VBox/ButtonRow/BtnReset"
@onready var btn_clear: Button = $"UILayer/Panel/VBox/ButtonRow/BtnClear"
@onready var btn_pause: Button = $"UILayer/Panel/VBox/ButtonRow/BtnPause"
@onready var btn_create: Button = $"UILayer/Panel/VBox/BrushModeRow/BtnCreate"
@onready var btn_erase: Button = $"UILayer/Panel/VBox/BrushModeRow/BtnErase"

# Checkboxes
@onready var chk_floor: CheckBox = $"UILayer/Panel/VBox/ChkFloor"
@onready var chk_waste: CheckBox = $"UILayer/Panel/VBox/ChkWaste"

func _ready() -> void:
	# Initialize slider values from simulation
	_sync_ui_from_simulation()
	
	# Connect signals
	slider_dt.value_changed.connect(_on_dt_changed)
	slider_r.value_changed.connect(_on_r_changed)
	slider_mutation.value_changed.connect(_on_mutation_changed)
	slider_inertia.value_changed.connect(_on_inertia_changed)
	slider_chemotaxis.value_changed.connect(_on_chemotaxis_changed)
	slider_gravity.value_changed.connect(_on_gravity_changed)
	slider_friction.value_changed.connect(_on_friction_changed)
	slider_hue.value_changed.connect(_on_hue_changed)
	
	btn_reset.pressed.connect(_on_reset_pressed)
	btn_clear.pressed.connect(_on_clear_pressed)
	btn_pause.pressed.connect(_on_pause_pressed)
	btn_create.pressed.connect(_on_create_pressed)
	btn_erase.pressed.connect(_on_erase_pressed)
	
	chk_floor.toggled.connect(_on_floor_toggled)
	chk_waste.toggled.connect(_on_waste_toggled)

func _sync_ui_from_simulation() -> void:
	if simulation == null:
		return
	
	slider_dt.value = simulation.dt
	slider_r.value = simulation.kernel_radius
	slider_mutation.value = simulation.mutation_rate
	slider_inertia.value = simulation.inertia
	slider_chemotaxis.value = simulation.chemotaxis
	slider_gravity.value = simulation.gravity
	slider_friction.value = simulation.friction
	slider_hue.value = simulation.brush_hue
	
	chk_floor.button_pressed = simulation.floor_enabled
	chk_waste.button_pressed = simulation.show_waste
	
	_update_labels()

func _update_labels() -> void:
	lbl_dt.text = "Delta T: %.2f" % slider_dt.value
	lbl_r.text = "Kernel Radius: %d" % int(slider_r.value)
	lbl_mutation.text = "Mutation: %.3f" % slider_mutation.value
	lbl_inertia.text = "Inertia: %.2f" % slider_inertia.value
	lbl_chemotaxis.text = "Chemotaxis: %.1f" % slider_chemotaxis.value
	lbl_gravity.text = "Gravity: %.3f" % slider_gravity.value
	lbl_friction.text = "Friction: %.2f" % slider_friction.value
	lbl_hue.text = "Species Hue: %.2f" % slider_hue.value

# Slider callbacks
func _on_dt_changed(value: float) -> void:
	simulation.dt = value
	_update_labels()

func _on_r_changed(value: float) -> void:
	simulation.kernel_radius = int(value)
	_update_labels()

func _on_mutation_changed(value: float) -> void:
	simulation.mutation_rate = value
	_update_labels()

func _on_inertia_changed(value: float) -> void:
	simulation.inertia = value
	_update_labels()

func _on_chemotaxis_changed(value: float) -> void:
	simulation.chemotaxis = value
	_update_labels()

func _on_gravity_changed(value: float) -> void:
	simulation.gravity = value
	_update_labels()

func _on_friction_changed(value: float) -> void:
	simulation.friction = value
	_update_labels()

func _on_hue_changed(value: float) -> void:
	simulation.brush_hue = value
	_update_labels()

# Button callbacks
func _on_reset_pressed() -> void:
	simulation.reset()

func _on_clear_pressed() -> void:
	simulation.clear()

func _on_pause_pressed() -> void:
	simulation.toggle_pause()
	btn_pause.text = "RESUME" if simulation.paused else "PAUSE"

func _on_create_pressed() -> void:
	simulation.brush_mode = 1

func _on_erase_pressed() -> void:
	simulation.brush_mode = -1

# Checkbox callbacks
func _on_floor_toggled(pressed: bool) -> void:
	simulation.floor_enabled = pressed

func _on_waste_toggled(pressed: bool) -> void:
	simulation.show_waste = pressed
