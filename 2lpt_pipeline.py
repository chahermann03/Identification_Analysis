#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pySim_lib as pysim
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import subprocess


# ----------------------------------------------------- parameters ---------------------------------------------------
n = 10
b = 800
set = "L800-N512"
# --------------------------------------------------------------------------------------------------------------------

h = 0.67
box = b * h
plt.rcParams.update({"text.usetex": True})

def voidfinder_reader(name):
    voids = np.loadtxt(name)
    radius = voids[:, 0] 
    #radius = voids[:, 4]        
    coords = voids[:, 1:4]      
    return radius, coords


# In[ ]:


#delete small voids
#do all files have same number of particles? YES!
example = np.loadtxt(f"/u/chahermann/Sparkling/input/2lpt_input/{set}_set1_2lpt_run0000.dat")
N_particles = example.shape[0]
print("Number of particles:", N_particles)
V_box = box**3
n_m = N_particles / V_box
print("Number density:", n_m)
r_min = (V_box / N_particles)**(1/3) * 2.5

void_data_incl_small = {}
void_data_dict = {}

num_files = min(len(void_data_incl_small), n)  # Ensures we don't exceed available data

for i in range(0, n):
    filename = f"{set}_set1_2lpt_run{i:04}"
    void_key = f"void_data_incl_small_{filename}"

    void_data_incl_small[void_key] = voidfinder_reader(f'/u/chahermann/Sparkling/output/2lpt_output/void_{filename}.dat')
    #void_data_incl_small[void_key] = voidfinder_reader(f'/u/chahermann/Revolver/output/2lpt_output/{filename}/zobov-Revolver_Voids_cat.txt')

    radii_incl_small, coords_incl_small = void_data_incl_small[void_key]

    allowed = radii_incl_small > r_min  

    radii = radii_incl_small[allowed]
    coords_voids = coords_incl_small[allowed]

    void_data = np.column_stack((radii, coords_voids))
    void_data_dict[f"void_data_{filename}"] = void_data
    #print(f"Number of all voids for {filename}:", len((radii_incl_small)))
    #print(f"Number of filtered voids for {filename}:", len(void_data))


# In[ ]:


import matplotlib.patches as patches
from matplotlib.lines import Line2D


#show 4 files for comparison
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 11), dpi=100)
#fig.suptitle("$ \mathrm{Void \ And \ DM \ Distribution \ Comparison \ For \ The \ 2lpt \ Data} $", fontsize=20, y=0.95)

z_min = 0
z_max = 5
filename = f"{set}_set1_2lpt_run0000"

void_key = f"void_data_incl_small_{filename}"
radii_incl_small, coords_incl_small = void_data_incl_small[void_key]

data_particles = np.loadtxt(f'/u/chahermann/Sparkling/input/2lpt_input/{filename}.dat')
positions = data_particles[:, 0:3]
velocities = data_particles[:, 3:6]
ax.set_facecolor('black')    
mask = (coords_incl_small[:, 2] >= z_min) & (coords_incl_small[:, 2] <= z_max)
ax.scatter(coords_incl_small[mask, 0]/h, coords_incl_small[mask, 1]/h, s=25, c='darkorange', label='$ \mathrm{Voids}', edgecolors='darkorange', linewidths=1., marker='x', zorder=3)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for x, y, r in zip(coords_incl_small[mask, 0]/h, coords_incl_small[mask, 1]/h, radii_incl_small[mask]/h):
    circle = patches.Circle((x, y), r, color='darkorange', fill=False, linewidth=1., alpha=1, zorder=3)
    ax.add_patch(circle)

# Plot dark matter particles in the (x, y) plane
mask_dm = (positions[:, 2] >= z_min) & (positions[:, 2] <= z_max)
ax.scatter(positions[mask_dm, 0]/h, positions[mask_dm, 1]/h, s=1, alpha=0.3, c='cyan', label='$ \mathrm{DM}$', zorder=1)
for spine in ax.spines.values():
    spine.set_edgecolor('white')

custom_legend = [ Line2D([0], [0], marker='x', color='darkorange', linestyle='None', markersize=10, label='$\mathrm{Voids}$', linewidth=1.5), Line2D([0], [0], marker='o', color='cyan', linestyle='None', markersize=7, alpha=1, label='$\mathrm{DM}$')]
legend = ax.legend(handles=custom_legend, fontsize=28, facecolor='black', edgecolor='white', labelcolor='white', framealpha=1, loc='lower left')
legend.get_frame().set_linewidth(2)

ax.tick_params(axis='both', which='both', direction='in', labelsize=26, color='white', length=10)
ax.set_xlabel("$X$ $[\mathrm{Mpc}]$", fontsize=28)
ax.set_ylabel("$Y$ $[\mathrm{Mpc}]$", fontsize=28)
ax.set_aspect('equal')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
# fig.savefig(f"/u/chahermann/2LPT_results/Plots/sparkling_spat_2lpt_{set}.pdf")
fig.savefig(f"/u/chahermann/2LPT_results/Plots/sparkling_spat_2lpt_{set}.png", dpi=100, bbox_inches='tight', pad_inches=0)


# In[ ]:


import itertools

#void size function
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 7), dpi=100)
#collect all void radii
all_2lpt_radii_incl_small = []
for i in range(0, n):
    filename = f"{set}_set1_2lpt_run{i:04}"
    void_key = f"void_data_incl_small_{filename}"
    radii_incl_small, _ = void_data_incl_small[void_key]

    if isinstance(radii_incl_small, (list, np.ndarray)):  
        all_2lpt_radii_incl_small.append(radii_incl_small)
    else:
        all_2lpt_radii_incl_small.append([radii_incl_small]) 
        
all_2lpt_radii_incl_small = list(itertools.chain.from_iterable(all_2lpt_radii_incl_small))

bins = np.logspace(np.log10(min(all_2lpt_radii_incl_small)), np.log10(max(all_2lpt_radii_incl_small)), 20)
print(min(all_2lpt_radii_incl_small))


hist_2lpt, bin_edges_2lpt= np.histogram(all_2lpt_radii_incl_small, bins=bins)
bin_centers = np.sqrt(bin_edges_2lpt[:-1] * bin_edges_2lpt[1:])
errors_2lpt = np.sqrt(hist_2lpt/(n-1))
ax.errorbar(bin_centers, hist_2lpt/(n-1), yerr=errors_2lpt, fmt='o', color='green', linestyle='-', capsize=5, markersize=3, label='$2LPT$')

ax.set_xscale('log')
ax.set_yscale('log')

ax.get_yaxis().set_tick_params(which='both', direction='in', labelsize=15)
ax.get_xaxis().set_tick_params(which='both', direction='in', labelsize=15)

ticks = np.logspace(np.log10(min(all_2lpt_radii_incl_small)), np.log10(max(all_2lpt_radii_incl_small)), 3)
ticks = np.append(ticks, r_min)  
ax.set_xticks(ticks)
ax.set_xticks([], minor=True)  # delete Minor-Ticks, falls nötig
ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}".rstrip('0').rstrip('.')))  

ax.axvline(x=r_min, color='red', linestyle='--', linewidth=2, label='$Threshold$')
plt.legend(fontsize = 18)

plt.xlabel('$Radii$  \ $(h^{-1} \ \mathrm{Mpc})$', fontsize = 18)
plt.ylabel('$Number \ Of \ Voids$', fontsize = 18)
plt.title('$Continuous \ Void \ Size \ Function \ 2LPT$', fontsize = 20)
fig.savefig(f"/u/chahermann/2LPT_results/Plots/sparkling_VSF_2lpt_means_{set}.pdf")


# In[ ]:


#plot histogram of particle velocities
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 7), dpi=100)
all_velocities = []
for i in range(0, n):
    filename = f"{set}_set1_2lpt_run{i:04}"
    particles_data = np.loadtxt(f"/u/chahermann/Sparkling/input/2lpt_input/{filename}.dat")
    x_velo = particles_data[:,3]
    all_velocities.append(x_velo)

x_min, x_max = -2000, 2000
bins = np.linspace(x_min, x_max, 31)
result = ax.hist(x_velo, bins=bins, color='blue', alpha=0.7, label='$X-Velo$')
plt.legend(fontsize = 18, framealpha = 1)
plt.xlim(x_min, x_max)

ax.get_yaxis().set_tick_params(which='both', direction='in', labelsize=15)
ax.get_xaxis().set_tick_params(which='both', direction='in', labelsize=15)
plt.xlabel("$v_X$", fontsize=18)
plt.ylabel("$Number \ Of \ Particles$", fontsize=18)
plt.title("$2LPT \ Velocity \ Distribution$", fontsize=18)


# In[ ]:


#integrated density profile
n_m = N_particles/V_box
R_max = np.max(all_2lpt_radii_incl_small)
print("R_max=", R_max)
print("R_min=", r_min)
all_delta_profiles_groups = {1: [], 2: [], 3: [], 4: []} 
all_radii_groups = {1: [], 2: [], 3: [], 4: []}

all_voids_sorted = []  #one file for all sorted voids together
sorted_voids_per_file = {} #updated dictionary with seperated but sorted files
#sort voids
for key, value in void_data_dict.items():
    sorted_void_arrays = sorted(value, key=lambda x: x[0])  
    sorted_voids_per_file[key] = np.array(sorted_void_arrays)
    all_voids_sorted.extend(sorted_void_arrays)

all_voids_sorted = np.array(sorted(all_voids_sorted, key=lambda x: x[0]))
all_radii = all_voids_sorted[:, 0]

group_limits = np.percentile(all_radii, [25, 50, 75, 100])
print("Group limits:", group_limits)

# Bins wie im funktionierenden Code
scaled_bins = np.linspace(0, 4, 31)
scaled_bin_centers = (scaled_bins[:-1] + scaled_bins[1:]) / 2

# Ergebnis-Container
all_delta_profiles_groups = {1: [], 2: [], 3: [], 4: []}
all_radii_groups = {1: [], 2: [], 3: [], 4: []}

# Schleife über Files
for i in range(0, n):
    filename = f"{set}_set1_2lpt_run{i:04}"
    key = f"void_data_{filename}"
    voids = sorted_voids_per_file[key]

    delta_profiles_groups = {1: [], 2: [], 3: [], 4: []}
    radii_groups = {1: [], 2: [], 3: [], 4: []}

    particles_data = np.loadtxt(f"/u/chahermann/Sparkling/input/2lpt_input/{filename}.dat")
    positions = particles_data[:, :3]
    
    tree = cKDTree(positions, boxsize=box + 1e-13)

    for R_v_i, center_i in zip(voids[:, 0], voids[:, 1:4]):
        if R_v_i < r_min:
            continue  # skip small voids, just in case...
        if r_min <= R_v_i < group_limits[0]:
            group = 1
        elif group_limits[0] <= R_v_i < group_limits[1]:
            group = 2
        elif group_limits[1] <= R_v_i < group_limits[2]:
            group = 3
        else:
            group = 4

        # physical bins
        bins = scaled_bin_centers * R_v_i
        density_profile = []

        for r in bins:
            indices = tree.query_ball_point(center_i, r)
            n_particles = len(indices)
            density = n_particles / ((4/3) * np.pi * r**3)
            density_profile.append(density)

        density_profile = np.array(density_profile)
        delta_profile = density_profile / n_m - 1

        delta_profiles_groups[group].append(delta_profile)
        radii_groups[group].append(R_v_i)

    #save each file
    np.save(f"/u/chahermann/2LPT_results/Numpy_files/{filename}_integrated_delta_profiles_population_spaced_groups.npy", (delta_profiles_groups, radii_groups))
    #np.save(f"/u/chahermann/2LPT_results/Numpy_files/rev_{filename}_integrated_delta_profiles_population_spaced_groups.npy", (delta_profiles_groups, radii_groups))

    #save for all files
    for group in delta_profiles_groups:
        all_delta_profiles_groups[group].extend(delta_profiles_groups[group])
        all_radii_groups[group].extend(radii_groups[group])

np.save(f"/u/chahermann/2LPT_results/Numpy_files/all_integrated_delta_profiles_population_spaced_groups_{set}.npy", (all_delta_profiles_groups, all_radii_groups))
#np.save("/u/chahermann/2LPT_results/Numpy_files/rev_all_integrated_delta_profiles_population_spaced_groups.npy", (all_delta_profiles_groups, all_radii_groups))


# In[ ]:


colors = {1: "#2e6c2f", 2: "#a8e6a1", 3: "#f7a56a", 4: "#f07a7a"}
alphas = {1: 0.1, 2: 0.3, 3: 0.4, 4: 0.06}
order = [1, 2, 3, 4]
limits = [r_min] + [float(i) for i in group_limits]
limits = [round(float(i), 1) for i in limits]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 15), dpi=100)
fig.suptitle("Integrated Void Density Profiles", fontsize=30, y=0.95)
axes = axes.flatten()

dummy_handles = []
for group in sorted(delta_profiles_groups.keys()):
    line, = ax.plot([], [], color=colors[group], label=f'$[{limits[group-1]}, \ {limits[group]}]$'+' $h^{-1} \ \mathrm{Mpc}$')
    dummy_handles.append(line)


for i, ax in enumerate(axes[:4], start=1):
    filename = f"{set}_set1_2lpt_run{i:04}"
    delta_profiles_groups, radii_groups = np.load(f"/u/chahermann/2LPT_results/Numpy_files/{filename}_integrated_delta_profiles_population_spaced_groups.npy", allow_pickle=True)
    #delta_profiles_groups, radii_groups = np.load(f"/u/chahermann/2LPT_results/Numpy_files/rev_{filename}_integrated_delta_profiles_population_spaced_groups.npy", allow_pickle=True)
    for group in delta_profiles_groups:
        for delta_profile in delta_profiles_groups[group]:
            ax.plot(scaled_bin_centers, delta_profile, marker='None', linestyle='-', color=colors[group], alpha=alphas[group], zorder = order[group-1])
            
    ax.set_ylim(-1.1, 2)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=20)
    ax.set_xlabel("$R/R_v$", fontsize=20)
    ax.set_ylabel("$\Delta(R)$", fontsize=20)
    ax.set_title(rf"$Void \ Density \ Profile \ For \ 2LPT{i}$", fontsize=23)
    ax.legend(fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axhline(y=0, color='k', ls='-', zorder=5, lw=0.5)
    #ax.axvline(x=1, color='k', ls='--', zorder=5)
    #ax.axhline(y=-0.8, color='k', ls='--', zorder=5)

handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles=dummy_handles, fontsize=20, framealpha = 1, loc='upper right', title="$Group \ Limits$", title_fontsize=18)


if len(axes) > 5:
    axes[5].axis('off')
    axes[4].axis('off')

fig.subplots_adjust(wspace=0.15, hspace=0.25)
plt.tight_layout()
fig.savefig(f"/u/chahermann/2LPT_results/Plots/sparkling_IDP_2lpt_{set}.pdf")


# In[ ]:


#means
#combine all profiles for mean
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 7), dpi=100)
plt.ylim(-1.1, 1)

#mean for each bin
for group in delta_profiles_groups:
    group_delta_profiles = np.array(delta_profiles_groups[group])
    mean_profile = np.mean(group_delta_profiles, axis=0)
    std_profile = np.std(group_delta_profiles, axis=0) / np.sqrt(len(group_delta_profiles))

    ax.errorbar(scaled_bin_centers, mean_profile, yerr=std_profile, marker='o', linestyle='-', color=colors[group], label=f'Group {group}', linewidth=1.5, markersize=1, capsize=3)

#overall mean
all_delta_profiles = []
for group in delta_profiles_groups:
    for delta_profile in delta_profiles_groups[group]:
        all_delta_profiles.append(delta_profile)
all_delta_profiles = np.array(all_delta_profiles)

#create mean
mean_delta_profile = np.mean(all_delta_profiles, axis=0)
std_delta_profile = np.std(all_delta_profiles, axis=0) / np.sqrt(len(all_delta_profiles))

#plot mean
ax.errorbar(scaled_bin_centers, mean_delta_profile, yerr=std_delta_profile, marker='o', linestyle='-', color='black', label= 'Mean', linewidth=2, markersize=2)
ax.get_yaxis().set_tick_params(which='both', direction='in', labelsize=15)
ax.get_xaxis().set_tick_params(which='both', direction='in', labelsize=15)
ax.axhline(y=0, color='k', ls='-', zorder=1, lw=0.5)
ax.axvline(x=1, color='k', ls=':')
ax.axhline(y=-0.7, color='k', ls=':')
plt.title('Integrated Density Profile Means 2LPT', fontsize=22)
plt.xlabel('$R/R_v$', fontsize=22)
plt.ylabel('$\Delta(R)$', fontsize=22)
ax.legend(handles=handles, fontsize=20)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=dummy_handles, fontsize=18, framealpha=1, loc='upper right', title="$Group Limits$", title_fontsize=18)
fig.savefig(f"/u/chahermann/2LPT_results/Plots/sparkling_IDP_2lpt_means_{set}.pdf")


# In[ ]:


#Differential density profile
def apply_pbc(positions, box_size):
    return np.mod(positions, box_size)

def minimum_image(vecs, box_size):
    return np.where(np.abs(vecs) > box_size / 2, vecs - np.sign(vecs) * box_size, vecs)

# Setup
scaled_bins = np.linspace(0, 4, 31)
scaled_bin_centers = (scaled_bins[:-1] + scaled_bins[1:]) / 2

all_voids_sorted = []
sorted_voids_per_file = {}

for key, value in void_data_dict.items():
    sorted_void_arrays = sorted(value, key=lambda x: x[0])  
    sorted_voids_per_file[key] = np.array(sorted_void_arrays)
    all_voids_sorted.extend(sorted_void_arrays)

all_voids_sorted = np.array(sorted(all_voids_sorted, key=lambda x: x[0]))
all_radii = all_voids_sorted[:, 0]
group_limits = np.percentile(all_radii, [25, 50, 75, 100])
print("Group limits:", group_limits)

# Ergebniscontainer
all_delta_profiles_groups = {1: [], 2: [], 3: [], 4: []}
all_radii_groups = {1: [], 2: [], 3: [], 4: []}

for i in range(0, n):
    filename = f"{set}_set1_2lpt_run{i:04}"
    key = f"void_data_{filename}"
    voids = sorted_voids_per_file[key]

    delta_profiles_groups = {1: [], 2: [], 3: [], 4: []}
    radii_groups = {1: [], 2: [], 3: [], 4: []}

    particles_data = np.loadtxt(f"/u/chahermann/Sparkling/input/2lpt_input/{filename}.dat")
    positions = particles_data[:, :3]
    positions = apply_pbc(positions, box)
    tree = cKDTree(positions, boxsize=box + 1e-13)

    for R_v_i, center_i in zip(voids[:, 0], voids[:, 1:4]):
        if R_v_i < r_min:
            continue

        if r_min <= R_v_i < group_limits[0]:
            group = 1
        elif group_limits[0] <= R_v_i < group_limits[1]:
            group = 2
        elif group_limits[1] <= R_v_i < group_limits[2]:
            group = 3
        else:
            group = 4

        
        indices = tree.query_ball_point(center_i, 4 * R_v_i)
        if len(indices) == 0:
            continue

        coords = positions[indices]
        vecs = coords - center_i
        vecs = minimum_image(vecs, box)  #PBC here!
        dists = np.linalg.norm(vecs, axis=1)

        shell_edges = np.linspace(0, 4 * R_v_i, 31)
        density_profile = []

        for j in range(len(shell_edges) - 1):
            in_shell = (dists >= shell_edges[j]) & (dists < shell_edges[j + 1])
            if np.any(in_shell):
                n_particles = np.sum(in_shell)
                volume = (4/3) * np.pi * (shell_edges[j+1]**3 - shell_edges[j]**3)
                density = n_particles / volume
                density_profile.append(density)
            else:
                density_profile.append(0)

        density_profile = np.array(density_profile)
        delta_profile = density_profile / n_m - 1

        delta_profiles_groups[group].append(delta_profile)
        radii_groups[group].append(R_v_i)

    #save each file
    np.save(
        f"/u/chahermann/2LPT_results/Numpy_files/{filename}_differential_delta_profiles_equipopulated_groups.npy",
        #f"/u/chahermann/2LPT_results/Numpy_files/rev_{filename}_differential_delta_profiles_equipopulated_groups.npy",

        (delta_profiles_groups, radii_groups))

    for group in delta_profiles_groups:
        all_delta_profiles_groups[group].extend(delta_profiles_groups[group])
        all_radii_groups[group].extend(radii_groups[group])
#save for all files
np.save(f"/u/chahermann/2LPT_results/Numpy_files/all_differential_delta_profiles_equipopulated_groups_{set}.npy", (all_delta_profiles_groups, all_radii_groups))
#np.save("/u/chahermann/2LPT_results/Numpy_files/rev_all_differential_delta_profiles_equipopulated_groups.npy", (all_delta_profiles_groups, all_radii_groups))


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 15), dpi=100)
fig.suptitle("$ \mathrm{Differential \ Void \ Density \ Profiles} $", fontsize=30, y=0.95)
axes = axes.flatten()

for i, ax in enumerate(axes[:4], start=1):
    filename = f"{set}_set1_2lpt_run{i:04}"
    delta_profiles_groups, radii_groups = np.load(f"/u/chahermann/2LPT_results/Numpy_files/{filename}_differential_delta_profiles_equipopulated_groups.npy", allow_pickle=True)
    #delta_profiles_groups, radii_groups = np.load(f"/u/chahermann/2LPT_results/Numpy_files/rev_{filename}_differential_delta_profiles_equipopulated_groups.npy", allow_pickle=True)

    for group in delta_profiles_groups:
        for delta_profile in delta_profiles_groups[group]:
            ax.plot(scaled_bin_centers, delta_profile, marker='None', linestyle='-', color=colors[group], alpha=alphas[group], zorder = order[group-1])
            
    ax.set_ylim(-1.1, 2)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=20)
    ax.set_xlabel("$R/R_v$", fontsize=20)
    ax.set_ylabel("$\Delta(R)$", fontsize=20)
    ax.set_title(rf"$Void \ Density \ Profile \ For \ 2LPT{i}$", fontsize=23)
    ax.legend(fontsize=20)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axhline(y=0, color='k', ls='-', zorder=5, lw=0.5)
    #ax.axvline(x=1, color='k', ls='--', zorder=5)
   #ax.axhline(y=-0.8, color='k', ls='--', zorder = 5)

handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles=dummy_handles, fontsize=20, framealpha = 1, loc='upper right', title="$Group Limits$", title_fontsize=18)


if len(axes) > 5:
    axes[5].axis('off')
    axes[5].axis('off')

fig.subplots_adjust(wspace=0.15, hspace=0.25)
plt.tight_layout()




# In[ ]:


#means
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 7), dpi=100)
plt.ylim(-1.1, 1)

dummy_handles = []
for group in sorted(delta_profiles_groups.keys()):
    line, = ax.plot([], [], color=colors[group], label=f'$[{limits[group-1]}, \ {limits[group]}]$'+' $h^{-1} \ \mathrm{Mpc}$')
    dummy_handles.append(line)
all_delta_profiles_groups, all_radii_groups = np.load(f"/u/chahermann/2LPT_results/Numpy_files/all_differential_delta_profiles_equipopulated_groups_{set}.npy", allow_pickle=True)
#all_delta_profiles_groups, all_radii_groups = np.load("/u/chahermann/2LPT_results/Numpy_files/rev_all_differential_delta_profiles_equipopulated_groups.npy", allow_pickle=True)

#mean for each bin
for group in all_delta_profiles_groups:
    group_delta_profiles = np.array(all_delta_profiles_groups[group])
    mean_profile = np.mean(group_delta_profiles, axis=0)
    std_profile = np.std(group_delta_profiles, axis=0) / np.sqrt(len(group_delta_profiles))

    ax.errorbar(scaled_bin_centers, mean_profile, yerr=std_profile, marker='o', linestyle='-', alpha = 1, color=colors[group], linewidth=1.5, markersize=2, capsize=3)

#overall mean
all_delta_profiles = []
for group in all_delta_profiles_groups:
    for delta_profile in all_delta_profiles_groups[group]:
        all_delta_profiles.append(delta_profile)
all_delta_profiles = np.array(all_delta_profiles)

#create mean
mean_delta_profile = np.mean(all_delta_profiles, axis=0)
std_delta_profile = np.std(all_delta_profiles, axis=0) / np.sqrt(len(all_delta_profiles))

#plot mean
ax.errorbar(scaled_bin_centers, mean_delta_profile, yerr=std_delta_profile, marker='o', linestyle='-', label = "Mean", color='black', linewidth=2, markersize=2)
ax.get_yaxis().set_tick_params(which='both', direction='in', labelsize=15)
ax.get_xaxis().set_tick_params(which='both', direction='in', labelsize=15)

ax.axhline(y=0, color='k', ls='-', zorder=1, lw=0.5)
#ax.axvline(x=1, color='k', ls='--')
#ax.axhline(y=-0.8, color='k', ls='--')
plt.title('$ \mathrm{Differential \ Void \ Density \ Profiles \ Means \ 2LPT} $', fontsize=20)
plt.xlabel('$R/R_v$', fontsize=22)
plt.ylabel('$\delta(R)$', fontsize=22)

handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles=dummy_handles, fontsize=20, framealpha = 1)
ax.legend(fontsize=18)



# In[ ]:


#velocity profiles

all_velocity_profiles_groups = {1: [], 2: [], 3: [], 4: []}
all_radii_groups = {1: [], 2: [], 3: [], 4: []}
scaled_bins = np.linspace(0, 4, 31)
scaled_bin_centers = (scaled_bins[:-1] + scaled_bins[1:]) / 2
for i in range(0, n):
    filename = f"{set}_set1_2lpt_run{i:04}"
    key = f"void_data_{filename}"
    voids = sorted_voids_per_file[key]

    velocity_profiles_groups = {1: [], 2: [], 3: [], 4: []}
    radii_groups = {1: [], 2: [], 3: [], 4: []}

    particles_data = np.loadtxt(f"/u/chahermann/Sparkling/input/2lpt_input/{filename}.dat")
    positions = particles_data[:, :3]
    velocities = particles_data[:, 3:6]
    positions = apply_pbc(positions, box)
    tree = cKDTree(positions, boxsize=box + 1e-13)

    for R_v_i, center_i in zip(voids[:, 0], voids[:, 1:4]):
        if R_v_i < r_min:
            continue

        if r_min <= R_v_i < group_limits[0]:
            group = 1
        elif group_limits[0] <= R_v_i < group_limits[1]:
            group = 2
        elif group_limits[1] <= R_v_i < group_limits[2]:
            group = 3
        else:
            group = 4

        
        indices = tree.query_ball_point(center_i, 4 * R_v_i)
        if len(indices) == 0:
            continue

        coords = positions[indices]
        vels = velocities[indices]
        vecs = coords - center_i
        vecs = minimum_image(vecs, box)  #PBC here!
        dists = np.linalg.norm(vecs, axis=1)
    
        v_rad = np.zeros_like(dists)
        valid = dists > 0
        v_rad[valid] = np.sum(vels[valid] * vecs[valid], axis=1) / dists[valid]

        #mean for each shell
        shell_edges = np.linspace(0, 4 * R_v_i, 31)
        shell_v_rad_mean = []

        for j in range(len(shell_edges) - 1):
            in_shell = (dists >= shell_edges[j]) & (dists < shell_edges[j + 1])
            if np.any(in_shell):
                mean_v = np.mean(v_rad[in_shell])
            else:
                mean_v = np.nan

            shell_v_rad_mean.append(mean_v)

        shell_v_rad_mean = np.array(shell_v_rad_mean)
        velocity_profiles_groups[group].append(shell_v_rad_mean)
        radii_groups[group].append(R_v_i)

    #save each file
    np.save(
        f"/u/chahermann/2LPT_results/Numpy_files/{filename}_velocity_profiles_equipopulated_groups.npy",
        #f"/u/chahermann/2LPT_results/Numpy_files/rev_{filename}_velocity_profiles_equipopulated_groups.npy",
        (velocity_profiles_groups, radii_groups))

    for group in velocity_profiles_groups:
        all_velocity_profiles_groups[group].extend(velocity_profiles_groups[group])
        all_radii_groups[group].extend(radii_groups[group])
#save for all files
np.save(f"/u/chahermann/2LPT_results/Numpy_files/all_velocity_profiles_equipopulated_groups_{set}.npy", (all_velocity_profiles_groups, all_radii_groups))
#np.save("/u/chahermann/2LPT_results/Numpy_files/rev_all_velocity_profiles_equipopulated_groups.npy", (all_velocity_profiles_groups, all_radii_groups))


# In[ ]:


alphas = {1: 0.07, 2: 0.07, 3: 0.07, 4: 0.05}
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 15), dpi=100)
axs = axs.flatten()

for i in range(1,5):
    filename = f"{set}_set1_2lpt_run{i:04}"
    velocity_profiles_groups, radii_groups = np.load(f"/u/chahermann/2LPT_results/Numpy_files/{filename}_velocity_profiles_equipopulated_groups.npy", allow_pickle=True)
    #velocity_profiles_groups, radii_groups = np.load(f"/u/chahermann/2LPT_results/Numpy_files/rev_{filename}_velocity_profiles_equipopulated_groups.npy", allow_pickle=True)
    ax = axs[i-1]
    for group in velocity_profiles_groups:
        for velocity_profile in velocity_profiles_groups[group]:
            ax.plot(scaled_bin_centers, velocity_profile, 
                    marker='None', linestyle='-', 
                    color=colors[group], alpha=alphas[group])
            
    ax.set_title(f"$Velocity \ Profile \ For \ 2LPT{i}$", fontsize=18)
    ax.set_xlabel('$R/R_v$', fontsize=18)
    ax.set_ylabel('$v_{rad}$', fontsize=18)
    ax.get_yaxis().set_tick_params(which='both', direction='in', labelsize=18)
    ax.get_xaxis().set_tick_params(which='both', direction='in', labelsize=18)

dummy_handles = []
for group in sorted(velocity_profiles_groups.keys()):
    line, = ax.plot([], [], color=colors[group], label=f'$[{limits[group-1]}, \ {limits[group]}]$'+' $h^{-1} \ \mathrm{Mpc}$')
    dummy_handles.append(line)

ax.legend(handles=dummy_handles, fontsize=18, loc = 'upper right', title="$Group \ Limits$", title_fontsize=18)


fig.suptitle("$ \mathrm{Void \ Velocity \ Profiles \ 2LPT} $", fontsize=30)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
fig.savefig(f"/u/chahermann/2LPT_results/Plots/sparkling_VP_2lpt_{set}.pdf")


# In[ ]:


#means
all_velocity_profiles_groups, all_radii_groups = np.load(f"/u/chahermann/2LPT_results/Numpy_files/all_velocity_profiles_equipopulated_groups_{set}.npy", allow_pickle=True)
#all_velocity_profiles_groups, all_radii_groups = np.load("/u/chahermann/2LPT_results/Numpy_files/rev_all_velocity_profiles_equipopulated_groups.npy", allow_pickle=True)
fig, ax = plt.subplots(figsize=(10, 7), dpi=100)

mean_profiles = {}
for group in all_velocity_profiles_groups:
    all_profiles = []
    for velocity_profile in velocity_profiles_groups[group]:
        all_profiles.append(velocity_profile)
    if all_profiles:
        all_profiles = np.array(all_profiles)  
        mean_profiles[group] = np.nanmean(all_profiles, axis=0)  # ignore NaNs
        std_profile = np.nanstd(all_profiles, axis=0) / np.sqrt(len(all_profiles))

        #plot mean of each group
        ax.errorbar(scaled_bin_centers, mean_profiles[group], yerr=std_profile, marker='o', color=colors[group], linewidth=1.5, label=f'$[{limits[group-1]}, \ {limits[group]}]$'+' $h^{-1} \ \mathrm{Mpc}$', capsize=2, markersize=2)

    group_velocity_profiles = np.array(all_velocity_profiles_groups[group])
    mean_profile = np.nanmean(group_velocity_profiles, axis=0)
    std_profile = np.nanstd(group_velocity_profiles, axis=0) / np.sqrt(np.sum(~np.isnan(group_velocity_profiles), axis=0))

all_void_profiles = []
for group in all_velocity_profiles_groups:
    for velocity_profile in velocity_profiles_groups[group]:
        all_void_profiles.append(velocity_profile)

if all_void_profiles:
    all_void_profiles = np.array(all_void_profiles)
    global_mean_profile = np.nanmean(all_void_profiles, axis=0)  # ignore NaNs
    std_profile = np.nanstd(all_void_profiles, axis=0) / np.sqrt(len(all_void_profiles))
    ax.errorbar(scaled_bin_centers, global_mean_profile, yerr=std_profile, color='black', linewidth=2, linestyle='-', label=f'$[{limits[group-1]}, \ {limits[group]}]$'+' $h^{-1} \ \mathrm{Mpc}$', capsize =2)

ax.axhline(y=0, color='k', ls='-', lw=0.5)
ax.axvline(x=1, color='k', ls='--')
ax.get_yaxis().set_tick_params(which='both', direction='in', labelsize=15)
ax.get_xaxis().set_tick_params(which='both', direction='in', labelsize=15)

plt.title('$\mathrm{Velocity \ Profiles \ Means \ 2LPT}$', fontsize=20)
plt.xlabel(r'$R/R_v$', fontsize=22)
plt.ylabel(r'$v_{\mathrm{rad}}(R)$', fontsize=22)

ax.legend(handles=dummy_handles + [ax.lines[-1]], fontsize=18, framealpha=1)
plt.tight_layout()
fig.savefig(f"/u/chahermann/2LPT_results/Plots/sparkling_VP_2lpt_means_{set}.pdf")

