import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Set style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['svg.fonttype'] = 'none'

# Sample data (you'd replace with your actual data)
np.random.seed(42)
n_subjects = 20
total_attempts = np.random.randint(1651, 80000, n_subjects)
total_attempts = np.sort(total_attempts)
min_attempts = 1651
species = ['Rhesus' if i % 3 == 0 else 'Tonkean' for i in range(n_subjects)]

# Create figure with multiple subplots for different artistic approaches
fig = plt.figure(figsize=(20, 24))

# ============ 1. BUBBLE GRAVITATIONAL LAYOUT ============
ax1 = fig.add_subplot(3, 2, 1)

# Create spiral positions for bubbles
angles = np.linspace(0, 4*np.pi, n_subjects)
radii = np.linspace(0.5, 4, n_subjects)
x_pos = radii * np.cos(angles)
y_pos = radii * np.sin(angles)

# Bubble sizes based on total attempts
max_size = 2000
bubble_sizes = (total_attempts / total_attempts.max()) * max_size

for i, (x, y, size, attempts, spec) in enumerate(zip(x_pos, y_pos, bubble_sizes, total_attempts, species)):
    # Outer bubble (total attempts)
    color = 'steelblue' if spec == 'Rhesus' else 'purple'
    outer_circle = Circle((x, y), np.sqrt(size/np.pi), color=color, alpha=0.4, linewidth=2)
    ax1.add_patch(outer_circle)
    
    # Inner circle (retained data)
    inner_size = (min_attempts / total_attempts.max()) * max_size
    inner_circle = Circle((x, y), np.sqrt(inner_size/np.pi), color='darkgreen', alpha=0.8)
    ax1.add_patch(inner_circle)

ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)
ax1.set_aspect('equal')
ax1.set_title('Bubble Gravitational Layout\n(Outer = Total, Inner = Retained)', fontsize=14, fontweight='bold')
ax1.axis('off')

# ============ 2. TREE RING VISUALIZATION ============
ax2 = fig.add_subplot(3, 2, 2)

# Arrange subjects in a grid
grid_size = int(np.ceil(np.sqrt(n_subjects)))
positions = [(i % grid_size, i // grid_size) for i in range(n_subjects)]

for i, (pos, attempts, spec) in enumerate(zip(positions, total_attempts, species)):
    x, y = pos
    
    # Calculate ring thicknesses
    max_radius = 0.4
    total_radius = (attempts / total_attempts.max()) * max_radius
    retained_radius = (min_attempts / total_attempts.max()) * max_radius
    
    # Color by species
    outer_color = 'steelblue' if spec == 'Rhesus' else 'purple'
    
    # Draw rings (from outside to inside)
    n_rings = 8
    for ring in range(n_rings):
        ring_radius = total_radius * (1 - ring / n_rings)
        if ring_radius > retained_radius:
            # Discarded data rings
            circle = Circle((x, y), ring_radius, color=outer_color, 
                          alpha=0.3 + 0.1 * ring, linewidth=0.5, fill=False)
        else:
            # Retained data rings
            circle = Circle((x, y), ring_radius, color='darkgreen', 
                          alpha=0.6, linewidth=1, fill=False)
        ax2.add_patch(circle)
    
    # Fill center (retained core)
    center_circle = Circle((x, y), retained_radius, color='darkgreen', alpha=0.4)
    ax2.add_patch(center_circle)

ax2.set_xlim(-0.5, grid_size - 0.5)
ax2.set_ylim(-0.5, grid_size - 0.5)
ax2.set_aspect('equal')
ax2.set_title('Tree Ring Growth Pattern\n(Rings = Data Layers)', fontsize=14, fontweight='bold')
ax2.axis('off')

# ============ 3. LIQUID CONTAINER VISUALIZATION ============
ax3 = fig.add_subplot(3, 2, 3)

container_width = 0.8
spacing = 1.0
n_per_row = 8
rows = int(np.ceil(n_subjects / n_per_row))

for i, (attempts, spec) in enumerate(zip(total_attempts, species)):
    row = i // n_per_row
    col = i % n_per_row
    x = col * spacing
    y = row * spacing
    
    # Container outline
    container_height = 3
    container = Rectangle((x, y), container_width, container_height, 
                         fill=False, edgecolor='black', linewidth=2)
    ax3.add_patch(container)
    
    # Total liquid level
    total_level = (attempts / total_attempts.max()) * container_height
    total_color = 'steelblue' if spec == 'Rhesus' else 'purple'
    total_liquid = Rectangle((x, y), container_width, total_level, 
                           color=total_color, alpha=0.6)
    ax3.add_patch(total_liquid)
    
    # Retained liquid level
    retained_level = (min_attempts / total_attempts.max()) * container_height
    retained_liquid = Rectangle((x, y), container_width, retained_level, 
                              color='darkgreen', alpha=0.8)
    ax3.add_patch(retained_liquid)
    
    # Drainage line (shows what gets discarded)
    if total_level > retained_level:
        ax3.plot([x, x + container_width], [retained_level, retained_level], 
                'r--', linewidth=2, alpha=0.7)

ax3.set_xlim(-0.5, n_per_row * spacing)
ax3.set_ylim(-0.5, rows * spacing + 3)
ax3.set_title('Liquid Container Metaphor\n(Red line = Drainage level)', fontsize=14, fontweight='bold')
ax3.axis('off')

# ============ 4. DNA STRAND VISUALIZATION ============
ax4 = fig.add_subplot(3, 2, 4)

strand_length = 10
x_positions = np.linspace(0, strand_length, n_subjects)

for i, (x, attempts, spec) in enumerate(zip(x_positions, total_attempts, species)):
    # Strand thickness based on total attempts
    thickness = (attempts / total_attempts.max()) * 0.3 + 0.05
    retained_thickness = (min_attempts / total_attempts.max()) * 0.3 + 0.05
    
    # Colors
    total_color = 'steelblue' if spec == 'Rhesus' else 'purple'
    
    # Draw strand segments
    y_base = 0
    y_top = 2
    
    # Total strand (background)
    ax4.plot([x, x], [y_base, y_top], color=total_color, 
            linewidth=thickness*50, alpha=0.5, solid_capstyle='round')
    
    # Retained strand (foreground)
    ax4.plot([x, x], [y_base, y_top], color='darkgreen', 
            linewidth=retained_thickness*50, alpha=0.9, solid_capstyle='round')
    
    # Connection lines (helix effect)
    if i < n_subjects - 1:
        next_x = x_positions[i + 1]
        ax4.plot([x, next_x], [y_top, y_base], 'k-', alpha=0.2, linewidth=0.5)
        ax4.plot([x, next_x], [y_base, y_top], 'k-', alpha=0.2, linewidth=0.5)

ax4.set_xlim(-0.5, strand_length + 0.5)
ax4.set_ylim(-0.5, 2.5)
ax4.set_title('DNA Strand Metaphor\n(Thickness = Data Volume)', fontsize=14, fontweight='bold')
ax4.axis('off')

# ============ 5. SOUND WAVE VISUALIZATION ============
ax5 = fig.add_subplot(3, 2, 5)

x = np.linspace(0, 4*np.pi, 1000)
baseline = 0

for i, (attempts, spec) in enumerate(zip(total_attempts, species)):
    # Create wave with amplitude based on attempts
    amplitude = (attempts / total_attempts.max()) * 2
    retained_amplitude = (min_attempts / total_attempts.max()) * 2
    
    # Frequency based on position
    frequency = 2 + i * 0.1
    wave = amplitude * np.sin(frequency * x)
    retained_wave = retained_amplitude * np.sin(frequency * x)
    
    # Vertical offset for each subject
    offset = i * 0.3
    
    # Colors
    wave_color = 'steelblue' if spec == 'Rhesus' else 'purple'
    
    # Plot waves
    ax5.plot(x, wave + offset, color=wave_color, alpha=0.6, linewidth=1.5)
    ax5.fill_between(x, offset, wave + offset, color=wave_color, alpha=0.2)
    
    # Retained portion
    ax5.plot(x, retained_wave + offset, color='darkgreen', alpha=0.9, linewidth=2)
    ax5.fill_between(x, offset, retained_wave + offset, color='darkgreen', alpha=0.4)

ax5.set_xlim(0, 4*np.pi)
ax5.set_ylim(-0.5, n_subjects * 0.3 + 2)
ax5.set_title('Sound Wave Symphony\n(Amplitude = Data Volume)', fontsize=14, fontweight='bold')
ax5.set_xlabel('Time/Frequency')
ax5.axis('off')

# ============ 6. ABSTRACT GEOMETRIC PATTERN ============
ax6 = fig.add_subplot(3, 2, 6)

# Create a mandala-like pattern
center_x, center_y = 0, 0
angles = np.linspace(0, 2*np.pi, n_subjects, endpoint=False)

for i, (angle, attempts, spec) in enumerate(zip(angles, total_attempts, species)):
    # Radial distance based on attempts
    max_radius = 4
    total_radius = (attempts / total_attempts.max()) * max_radius
    retained_radius = (min_attempts / total_attempts.max()) * max_radius
    
    # End points
    total_x = center_x + total_radius * np.cos(angle)
    total_y = center_y + total_radius * np.sin(angle)
    retained_x = center_x + retained_radius * np.cos(angle)
    retained_y = center_y + retained_radius * np.sin(angle)
    
    # Colors
    total_color = 'steelblue' if spec == 'Rhesus' else 'purple'
    
    # Draw rays
    ax6.plot([center_x, total_x], [center_y, total_y], 
            color=total_color, linewidth=8, alpha=0.4, solid_capstyle='round')
    ax6.plot([center_x, retained_x], [center_y, retained_y], 
            color='darkgreen', linewidth=6, alpha=0.8, solid_capstyle='round')
    
    # Add geometric shapes at endpoints
    triangle = patches.RegularPolygon((total_x, total_y), 3, 
                                    radius=0.1, color=total_color, alpha=0.7)
    ax6.add_patch(triangle)

# Center circle
center_circle = Circle((center_x, center_y), 0.2, color='gold', alpha=0.9)
ax6.add_patch(center_circle)

ax6.set_xlim(-5, 5)
ax6.set_ylim(-5, 5)
ax6.set_aspect('equal')
ax6.set_title('Geometric Mandala\n(Ray length = Data Volume)', fontsize=14, fontweight='bold')
ax6.axis('off')

plt.tight_layout()
plt.show()

# Create a summary
print("Artistic Visualization Options:")
print("1. Bubble Layout: Natural size scaling, spiral arrangement")
print("2. Tree Rings: Growth metaphor, handles range well")
print("3. Liquid Containers: Intuitive 'filling/draining' concept")
print("4. DNA Strands: Scientific aesthetic, thickness encoding") 
print("5. Sound Waves: Musical metaphor, layered harmonics")
print("6. Geometric Mandala: Abstract art, radial symmetry")