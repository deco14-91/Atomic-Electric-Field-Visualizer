import numpy as np
from mayavi import mlab

# ================== User-configurable parameters ===============================================
density = 3          # Arrow spacing (higher = sparser)
arrow_len = 1.0      # Constant arrow length
arrow_thickness = 4  # Thickness of arrows
slice_coord = 20.0   # Real-space coordinate for slicing (e.g., in angstroms)
plane = 'YZ'         # Options: 'XY', 'XZ', 'YZ'

# ================= Color map: atomic number RGB color ==========================================
atom_colors = {
    1: (1.0, 1.0, 1.0),  # H - white
    3: (1.0, 0.4, 0.7),  # Li - pink
    6: (0.0, 1.0, 0.0),  # C - green
    7: (0.0, 0.0, 1.0),  # N - blue
    8: (1.0, 0.0, 0.0),  # O - red
    9: (0.0, 1.0, 1.0),  # F - cyan
    15: (1.0, 0.5, 0.0), # P - orange
    16: (1.0, 1.0, 0.0), # S - yellow
}
atom_radii = {
    1: 0.5,   # H
    3: 1.20,   # Li
    6: 1.20,   # C
    7: 1.20,   # N
    8: 1.20,   # O
    9: 1.20,   # F
    15: 2.00,  # P
    16: 1.20,  # S
}

# ================= Bond cutoff distances (in angstroms) by atomic number pairs ================
bond_cutoffs = {
    (3, 3): 6.8,
    (6, 6): 3.5,
    (16, 8): 3.5,
    (16, 7): 3.5,
    (6, 1): 2.4,
    (7, 6): 3.5,
    (16, 6): 4.0,
    (16, 9): 3.5,
    (6, 9): 3.5,
    (3, 8): 3.5,   #Li-O
    (3, 9): 4.3,   #Li-F
    (3, 16): 4.64, #Li-S
    # symmetric pairs
    (8, 16): 3.5,
    (7, 16): 3.5,
    (1, 6): 2.4,
    (6, 7): 3.5,
    (6, 16): 4.0,
    (9, 16): 3.5,
    (9, 6): 3.5,
    (8, 3): 3.5,
    (9, 3): 3.5,
    (16, 3): 3.5,
}

def read_cube_file(filename):
    with open(filename) as f:
        lines = f.readlines()

    origin_line = lines[2].split()
    natoms = int(origin_line[0])
    origin = np.array([float(x) for x in origin_line[1:]])

    nx, vx = int(lines[3].split()[0]), np.array(lines[3].split()[1:], dtype=float)
    ny, vy = int(lines[4].split()[0]), np.array(lines[4].split()[1:], dtype=float)
    nz, vz = int(lines[5].split()[0]), np.array(lines[5].split()[1:], dtype=float)

    atoms = []
    for i in range(6, 6 + abs(natoms)):
        parts = lines[i].split()
        atomic_number = int(parts[0])
        pos = np.array([float(p) for p in parts[2:]])
        atoms.append((atomic_number, pos))

    data_lines = lines[6 + abs(natoms):]
    data = np.array([float(val) for line in data_lines for val in line.split()])
    data = data.reshape((nx, ny, nz))

    x = np.linspace(0, vx[0] * nx, nx) + origin[0]
    y = np.linspace(0, vy[1] * ny, ny) + origin[1]
    z = np.linspace(0, vz[2] * nz, nz) + origin[2]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    return X, Y, Z, data, atoms

def compute_electric_field(X, Y, Z, V):
    dV_dx, dV_dy, dV_dz = np.gradient(
        V,
        X[1,0,0] - X[0,0,0],
        Y[0,1,0] - Y[0,0,0],
        Z[0,0,1] - Z[0,0,0]
    )
    Ex, Ey, Ez = -dV_dx, -dV_dy, -dV_dz
    return Ex, Ey, Ez

def real_to_index(coord, axis_coords):
    idx = (np.abs(axis_coords - coord)).argmin()
    return idx

def draw_bond(pos1, pos2, radius=0.1, color=(0.3, 0.3, 0.3)):
    mlab.plot3d(
        [pos1[0], pos2[0]],
        [pos1[1], pos2[1]],
        [pos1[2], pos2[2]],
        tube_radius=radius,
        color=color
    )

def visualize_field_slice(X, Y, Z, Ex, Ey, Ez, atoms, plane, slice_index):
    mlab.figure(size=(900, 700), bgcolor=(0.41, 0.41, 0.41))

    #========================================Draw bonds first==================================
    for i, (z1, pos1) in enumerate(atoms):
        for j, (z2, pos2) in enumerate(atoms):
            if j <= i:
                continue
            pair = (z1, z2) if (z1, z2) in bond_cutoffs else (z2, z1)
            if pair in bond_cutoffs:
                cutoff = bond_cutoffs[pair]
                distance = np.linalg.norm(pos1 - pos2)
                if distance < cutoff:
                    draw_bond(pos1, pos2)

    #================================================Draw atoms================================
    for atomic_number, pos in atoms:
        color = atom_colors.get(atomic_number, (0.5, 0.5, 0.5))
        radius = atom_radii.get(atomic_number, 0.4)
        mlab.points3d(
            pos[0], pos[1], pos[2],
            scale_factor=radius,
            color=color,
            resolution=20
        )

    #====================================Select field slice=====================================
    if plane == 'XY':
        xs = X[:, :, slice_index]
        ys = Y[:, :, slice_index]
        zs = Z[:, :, slice_index]
        u = Ex[:, :, slice_index]
        v = Ey[:, :, slice_index]
        w = Ez[:, :, slice_index] * 0

    elif plane == 'XZ':
        xs = X[:, slice_index, :]
        ys = Y[:, slice_index, :]
        zs = Z[:, slice_index, :]
        u = Ex[:, slice_index, :]
        v = Ey[:, slice_index, :] * 0
        w = Ez[:, slice_index, :]

    elif plane == 'YZ':
        xs = X[slice_index, :, :]
        ys = Y[slice_index, :, :]
        zs = Z[slice_index, :, :]
        u = Ex[slice_index, :, :] * 0
        v = Ey[slice_index, :, :]
        w = Ez[slice_index, :, :]

    else:
        raise ValueError("Invalid plane. Choose from 'XY', 'XZ', 'YZ'.")

    #==============================Subsample field============================================
    x_sub = xs[::density, ::density]
    y_sub = ys[::density, ::density]
    z_sub = zs[::density, ::density]
    u_sub = u[::density, ::density]
    v_sub = v[::density, ::density]
    w_sub = w[::density, ::density]

    #==============================Normalize direction vectors================================
    norms = np.sqrt(u_sub**2 + v_sub**2 + w_sub**2)
    norms[norms == 0] = 1  # prevent division by zero
    u_dir = u_sub / norms
    v_dir = v_sub / norms
    w_dir = w_sub / norms

    #=========================Shaft of the arrow (line mode)==================================
    mlab.quiver3d(
        x_sub,
        y_sub,
        z_sub,
        u_sub,
        v_sub,
        w_sub,
        line_width=arrow_thickness,
        mode='2ddash',  # or 'arrow', 'line'
        scale_factor=arrow_len * 0.5,
        scale_mode='none',
        color=(0, 0, 0)
    )

    #================= Cone tip of the arrow (placed at tip of vector)=======================
    tip_x = x_sub + u_dir * arrow_len * 0.5
    tip_y = y_sub + v_dir * arrow_len * 0.5
    tip_z = z_sub + w_dir * arrow_len * 0.5

    mlab.quiver3d(
        tip_x,
        tip_y,
        tip_z,
        u_dir,
        v_dir,
        w_dir,
        mode='cone',
        scale_factor=arrow_len * 0.7,
        scale_mode='none',
        color=(0, 0, 0)
    )
    
    #=====================================Camera setup=======================================
    scene = mlab.gcf().scene
    camera = scene.camera
    camera.parallel_projection = True

    if plane == 'XY':
        mlab.view(azimuth=0, elevation=90, distance='auto')
    elif plane == 'XZ':
        mlab.view(azimuth=0, elevation=0, distance='auto')
    elif plane == 'YZ':
        mlab.view(azimuth=90, elevation=0, distance='auto')

    mlab.title(f" ", size=0.4)
    
    #======================Add key press binding to save high-res transparent PNG============
    def save_highres(obj=None, evt=None):
        mlab.savefig("field_visualization.png", size=(1600, 1200), magnification=3)
        print("High-res image saved as 'field_visualization.png'")

    scene.interactor.add_observer("KeyPressEvent", lambda obj, evt: save_highres())
    
    mlab.show()

if __name__ == "__main__":
    cube_filename = "totesp.cub"  # Replace with your .cube file
    X, Y, Z, V, atoms = read_cube_file(cube_filename)
    Ex, Ey, Ez = compute_electric_field(X, Y, Z, V)

    if plane == 'XY':
        axis_vals = Z[0, 0, :]
    elif plane == 'XZ':
        axis_vals = Y[0, :, 0]
    elif plane == 'YZ':
        axis_vals = X[:, 0, 0]
    else:
        raise ValueError("Invalid plane. Choose from 'XY', 'XZ', 'YZ'.")

    slice_index = real_to_index(slice_coord, axis_vals)
    visualize_field_slice(X, Y, Z, Ex, Ey, Ez, atoms, plane=plane, slice_index=slice_index)
