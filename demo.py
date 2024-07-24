#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 18:22:23 2021

@author: Ruadhri McDougall
"""

import raytrace as ray
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# sns.set()
# sns.set_style('darkgrid')
# sns.set_context('paper', font_scale=1.5)

#%%
# =============================================================================
# Task 9 - Tracing Example Rays Through Spherical Surface
# =============================================================================
ax = plt.axes(projection='3d')

spher_location = 100e-3
curv = 30
ref_1 = 1
ref_2 = 1.5
output_loc = np.array([0,0,350e-3])
 
# create 5 rays to propagate through the spherical lens
ray1 = ray.Ray(np.array([0,0.02,0]), np.array([0,0,1]))
ray2 = ray.Ray(np.array([0,0.01,0]), np.array([0,0,1]))
ray3 = ray.Ray(np.array([0,0,0]), np.array([0,0,1]))
ray4 = ray.Ray(np.array([0,-0.01,0]), np.array([0,0,1]))
ray5 = ray.Ray(np.array([0,-0.02,0]), np.array([0,0,1]))

# create spherical lens and output plane
spher_lens = ray.SphericalRefraction(spher_location, curv, ref_1, ref_2, 0.025)
out = ray.OutputPlane(output_loc, y_len=1)

ray_lst = np.array([ray1, ray2, ray3, ray4, ray5])

plt.figure()
for i in range(len(ray_lst)):
    new_ray = ray_lst[i]
    spher_lens.propagate_ray(new_ray)
    out.propagate_ray(new_ray)
    new_ray.show2D()

spher_lens.show2D()
plt.xlabel('z (m)')
plt.ylabel('y (m)')
plt.title('Task 9 (2D)')
plt.tight_layout()
# plt.savefig('task9_2d.png', dpi=700)


plt.figure()
for i in range(len(ray_lst)):
    new_ray = ray_lst[i]
    new_ray.show3D(ax)

ax.set_title('Task 9 (3D)')
ax.set_xlabel('z (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('x (m)')
#ax.tight_layout()
#plt.savefig('task9_3d.png', dpi=700)

focal_len = ref_2 / (curv * (ref_2-ref_1))
focus_computed = ray.focal_point([spher_lens]) - spher_location

print('Theoretical Focal Length is %s'%(focal_len))
print('Computed Focus Length is %s'%(focus_computed))


#%%
# =============================================================================
# Task 11 - Simple Test Cases
# =============================================================================

# See testing.py for simple test cases as what I did additionally did not have
# a nice or relevant graphical output to be used in the report, or have a 
# graphical output at all.

#%%
# =============================================================================
# Task 12 - Bundle of Collimated Rays Through Spherical Surface
# =============================================================================
ax = plt.axes(projection='3d')

spher_location = 100e-3
curv = 30
ref_1 = 1
ref_2 = 1.5
output_loc = np.array([0,0,350e-3])

beam1 = ray.Beam(30, 0.005)

# # create spherical lens and output plane
spher = ray.SphericalRefraction(spher_location, curv, ref_1, ref_2, 0.0065)
output = ray.OutputPlane(output_loc, x_len=1, y_len=1)

spher.propagate_beam(beam1)
output.propagate_beam(beam1)

plt.figure()
beam1.show2D()
spher.show2D()
plt.xlabel('z (m)')
plt.ylabel('y (m)')
plt.title('Task 12 (2D)')
plt.tight_layout()
# plt.savefig('task12_2d.png', dpi=700)

plt.figure()
beam1.show3D(ax)
ax.set_title('Task 12 (3D)')
ax.set_xlabel('z (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('x (m)')
#ax.tight_layout()
#plt.savefig('task12_3d.png', dpi=700)



#%%
# =============================================================================
# Task 13 - Spot Diagrams for Bundle of Rays
# =============================================================================

spher_location_new = 100e-3
curv_new = 30
ref_1_new = 1
ref_2_new = 1.5

beam_new = ray.Beam(30, 0.005)

# # create spherical lens and output plane
spher_new = ray.SphericalRefraction(spher_location, curv_new, ref_1_new, ref_2_new, 0.02)

rms_size_beam1 = ray.spot_plot(beam_new, [spher_new])
print('The RMS radius for a bundle of rays is %sm'%rms_size_beam1)

plt.xlabel('x (m)')
plt.ylabel('y (m)')
#plt.title('Task 12 Spot Plot')
plt.tight_layout()
#plt.savefig('task13_spotplot.png', dpi=700)

#%%
# =============================================================================
# Task 15 - Ray Trajectories and Performance of Plano-Convex Lens Orientations
# =============================================================================

# plane side information
plane_location = 100e-3
plane_curv = 0
plane_ref_1 = 1
plane_ref_2 = 1.5168

# convex side information
con_location = plane_location + 5e-3
con_curv = -0.02e3
con_ref_1 = 1.5168
con_ref_2 = 1

beam1 = ray.Beam(30, 5e-3)

# create plano-convex singlet
plano_side1 = ray.SphericalRefraction(plane_location, plane_curv, plane_ref_1, plane_ref_2, 1e-2)
convex_side1 = ray.SphericalRefraction(con_location, con_curv, con_ref_1, con_ref_2, 1e-2)

system1 = [plano_side1, convex_side1]

plt.figure()
rms_size1 = ray.spot_plot(beam1, system1, label='P-C')

print('Plane first convex second gives focus of %sm'%(rms_size1))

# plane side information
plane_location = 105e-3
plane_curv = 0
plane_ref_1 = 1.5168
plane_ref_2 = 1

# convex side information
con_location = plane_location - 5e-3
con_curv = 0.02e3
con_ref_1 = 1
con_ref_2 = 1.5168

beam2 = ray.Beam(30, 5e-3)
beam3 = ray.Beam(90, 5e-3, arrange='square')

# # create plano-convex singlet
plano_side2 = ray.SphericalRefraction(plane_location, plane_curv, plane_ref_1, plane_ref_2, 1e-2)
convex_side2 = ray.SphericalRefraction(con_location, con_curv, con_ref_1, con_ref_2, 1e-2)

system2 = [convex_side2, plano_side2]

rms_size2 = ray.spot_plot(beam2, system2, label='C-P')
#plt.title('Spot Plot for Plano Convex Lens Orientations')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
# plt.savefig('task15_spotplot.png', dpi=700)

print('Convex first plane second gives focus of %sm'%(rms_size2))

plt.figure()
plt.scatter(beam1.vertices()[:,0,0], beam1.vertices()[:,0,1])
plt.title('Initial Beam in Circle Arrangement')
plt.xlabel('z (m)')
plt.ylabel('y (m)')
plt.tight_layout()
# plt.savefig('task15_2d_circ.png', dpi=700)

plt.figure()
plt.scatter(beam3.vertices()[:,0,0], beam3.vertices()[:,0,1])
plt.title('Initial Beam in Square Arrangement')
plt.xlabel('z (m)')
plt.ylabel('y (m)')
plt.tight_layout()
# plt.savefig('task15_2d_squar.png', dpi=700)


plt.figure()
beam1.show2D(style='line')
plano_side1.show2D()
convex_side1.show2D()
plt.title('Task 15 - Prop. to Focus (P-C)')
plt.xlabel('z (m)')
plt.ylabel('y (m)')
plt.tight_layout()
# plt.savefig('task15_2d_prop_pc.png', dpi=700)


plt.figure()
beam2.show2D()
plano_side2.show2D()
convex_side2.show2D()
plt.title('Task 15 - Prop. to Focus (C-P)')
plt.xlabel('z (m)')
plt.ylabel('y (m)')
plt.tight_layout()
# plt.savefig('task15_2d_prop_cp.png', dpi=700)

plt.figure()
ax1 = plt.axes(projection='3d')
beam1.show3D(ax1)
ax1.set_title('Task 15 - Prop. to Focus (P-C)')
ax1.set_xlabel('z (m)')
ax1.set_ylabel('y (m)')
ax1.set_zlabel('x (m)')
#plt.tight_layout()
# plt.savefig('task15_3d_prop_cp.png', dpi=700)

plt.figure()
ax2 = plt.axes(projection='3d')
beam2.show3D(ax2)
ax2.set_title('Task 15 - Prop. to Focus (C-P)')
ax2.set_xlabel('z (m)')
ax2.set_ylabel('y (m)')
ax2.set_zlabel('x (m)')
# plt.tight_layout()
# plt.savefig('task15_2=3d_prop_cp.png', dpi=700)


#%%
# =============================================================================
# Lens Optimisation
# =============================================================================

surf_1_loc = 100e-3
thick = 5e-3
surf_2_loc = surf_1_loc + thick
ref_out = 1
ref_in = 1.5168
ap_rad = 7e-3
num_measure = 10

beam_params = [30,5e-3,5]

lens_param = [surf_1_loc, thick, ref_out, ref_in, ap_rad]

curvature_guess = [31, 10]

result_min = ray.plano_convex_optimise(lens_param, curvature_guess, beam_params)
print()
print('The optimised first surface curvature is %s and the second %s'%(result_min[0], 
                                                                       result_min[1]))

curv1 = result_min[0]
curv2 = result_min[1]

lens1 = ray.SphericalRefraction(surf_1_loc, curv1, ref_out, ref_in, ap_rad)
lens2 = ray.SphericalRefraction(surf_2_loc, curv2, ref_in, ref_out, ap_rad)
sys = [lens1, lens2]
beam_test = ray.Beam(30, 5e-3)

rms_rad_min = ray.get_rms_size(beam_test, sys)
print('The minimised RMS radius is %s metres'%rms_rad_min)

plt.figure()
beam_test.show2D()
lens1.show2D()
lens2.show2D()
plt.title('Lens Optimisation Result')
plt.xlabel('z (m)')
plt.ylabel('y (m)')
plt.tight_layout()
#plt.savefig('optimisation_prop_2d.png', dpi=700)

'''
The code below repeats the above for a given number of measurements to calculate
mean curvatures and rms radius, and associated errors. It seems as though this 
has a negligible effect on the outcome however, and the error shows as being
zero in most cases due to being so small

Code has been commented out due to taking a while to run (~20s x num_measurements)
'''

# curv_ones = []
# curv_twos = []
# rms_rads = []
# for i in range(num_measure):
#     result_min = ray.plano_convex_optimise(lens_param, curvature_guess, beam_params)
#     curv1 = result_min[0]
#     curv2 = result_min[1]
#     beam_test = ray.Beam(30, 5e-3)
#     rms_rad_min = ray.get_rms_size(beam_test, sys)
#     curv_ones.append(curv1)
#     curv_twos.append(curv2)
#     rms_rads.append(rms_rad_min)

# mean_curv_one = np.mean(curv_ones)
# err_curv_one = np.std(curv_ones) / np.sqrt(num_measure)
# mean_curv_two = np.mean(curv_twos)
# err_curv_two = np.std(curv_twos) / np.sqrt(num_measure)
# mean_rms = np.mean(rms_rads)
# err_rms = np.std(rms_rads) / np.sqrt(num_measure)

