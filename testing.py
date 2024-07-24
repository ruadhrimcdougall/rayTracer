#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:06:55 2021

@author: Ruadhri McDougall
"""

import raytrace as ray
import numpy as np
import matplotlib.pyplot as plt

#%%
# =============================================================================
# Test 1 - Making sure I can create rays
# =============================================================================

# test_ray = Ray(start_pos = np.array([0,0.8,4]), start_dir = np.array([0,0,1]))
# 
# spher_len = SphericalRefraction(np.array([0,0,5]), 1, n_1=1, n_2=8/3, aperture_radius=0.9)
# 
# spher_len.propagate_ray(test_ray)
# print(test_ray.vertices())
# print(test_ray.all_ray_directions())
# =============================================================================


#%%
# =============================================================================
# Test 2 - single spherical refracting surface
# =============================================================================


# spher_location = np.array([0,0,100e-3])
# curv = 0
# ref_1 = 1
# ref_2 = 1.5
# output_loc = np.array([0,0,250e-3])
# 
# # create 5 rays to propagate through the spherical lens
# ray1 = ray.Ray(np.array([0,0.02,0]), np.array([0,0,1]))
# ray2 = ray.Ray(np.array([0,0.01,0]), np.array([0,0,1]))
# ray3 = ray.Ray(np.array([0,0,0]), np.array([0,0,1]))
# ray4 = ray.Ray(np.array([0,-0.01,0]), np.array([0,0,1]))
# ray5 = ray.Ray(np.array([0,-0.02,0]), np.array([0,0,1]))
# 
# # create spherical lens and output plane
# spher_lens = ray.SphericalRefraction(spher_location, curv, ref_1, ref_2, 0.04)
# out = ray.OutputPlane(output_loc, y_len=1)
# 
# ray_lst = np.array([ray1, ray2, ray3, ray4, ray5])
# z_vertices = []
# y_vertices = []
# 
# for i in range(len(ray_lst)):
#     new_ray = ray_lst[i]
#     spher_lens.propagate_ray(new_ray)
#     out.propagate_ray(new_ray)
#     points = new_ray.vertices()
#     z_vertices = points[:,2]
#     y_vertices = points[:,1]
#     #print(y_vertices)
#     #print(len(points))
#     #print(points)
#     #plt.figure()
#     plt.plot(z_vertices, y_vertices)
#     #print(points)
# 
# spher_lens.show2D()


#=============================================================================
# uncovered issue arrising from the intercept method in the SphericalRefraction
# class, where I was overwriting the class variable _surface_intercept to a 
# new value each time I use the method. This value should stay constant.

# also tested zero curvature and ironed out bugs associated with this.
# for example the way in which the surface normal of the lens was calculated
# based on a zero curvature had to be altered to give the right answer
# =============================================================================


#%%
# =============================================================================
# Test 3 - Testing the creation of beams from a series of rays
# =============================================================================

# 
# x = np.linspace(1,3,3)
# y = np.linspace(10,30,3)
# #z = np.linspace(100,1000,10)
# adder = np.array([1,2,3])
# 
# 
# all_xy = np.array(np.meshgrid(x, y))
# all_x = all_xy[0]
# all_y = all_xy[1]
# 
# mags = np.linalg.norm(all_xy, axis=0)
# 
# positions = []
# 
# for i in range(len(x)):
#     for j in range(len(x)):
#         #ray_position = np.array([ray_pos_x[x], ray_pos_y[y], 0])
#         #ray_position_mag = np.linalg.norm(ray_position)
#         
#         if mags[i,j] <= 10.2:
#             position = np.array([x[i], y[j], 0])
#             #ray_position = ray_position + start_pos
#             positions.append(position)
#             
# positions = np.array(positions)


#a = np.array(np.where(mags<10.1))
#a = np.where(mags<20)
#print(a)
#all_x = np.delete(all_xy, a[0], axis=0)

#x = np.delete(x, a)
#y = np.delete(y, a)
#z = np.delete(z, a)

#all_xyz = np.stack((x, y, z), axis=1)

#all_xyz = np.add(np.meshgrid(x, y, 0), 0)#np.array([1,1,1]))
#all_xyz = np.meshgrid(x, y)

#all_pos = all_xyz.T

#print(all_xyz)

#=============================================================================
# settled on using the np.meshgrid function to efficiently calculate the
# magnitude of the distance of the ray from the centre.

# will have to use for loops to create the ray anyway so makes most sense to
# calculate the ray position and initialise the ray within those for loops

# unfortunately I can't see a way around using nested for loops for initialising
# each ray, as I need to iterate over every x position for each y position.
# =============================================================================


#%%
# =============================================================================
# Test 4 - Beam class functionality
# =============================================================================

#ax = plt.axes(projection='3d')

# spher_location = np.array([0,0,100e-3])
# curv = 30
# ref_1 = 1
# ref_2 = 1.5
# output_loc = np.array([0,0,250e-3])
# 
# beam1 = ray.Beam(49, 0.002)
# 
# #for i in range(beam1.size()):
#     #print(beam1.all_rays()[i].termination())
# #print(beam1.size())
# 
# # # create spherical lens and output plane
# spher_lens = ray.SphericalRefraction(spher_location, curv, ref_1, ref_2, 0.0045)
# out = ray.OutputPlane(output_loc, x_len=0)
# 
# spher_lens.propagate_beam(beam1)
# out.propagate_beam(beam1)
# 
# #plt.figure()
# beam1.show2D()
# print(beam1.vertices()[:,:,1])
# spher_lens.show2D()
# 
# #for i in range(beam1.size()):
# #    print(beam1.all_rays()[i].termination())

#=============================================================================
# Found several bugs, for example the vertices function was only returning the
# first vertex, after which it would return zeros. Also there was an issue with a
# logic statement in the intercept function which resulted in half the rays in
# the beam being terminated. There was an issue with trying to plot
# terminated rays as I was not assigning them an equal number of vertices as
# other rays so there were broadcasting probelms.
# =============================================================================


#%%
# =============================================================================
# Test 5 - Raytracing Investigations
# =============================================================================

#ax = plt.axes(projection='3d')

# spher_location = np.array([0,0,100e-3])
# curv = 0.03e3
# ref_1 = 1
# ref_2 = 1.5
# focal_length = 1/(curv*(ref_2-1))
# focus = focal_length + spher_location[2] + (1/curv)
# output_loc = np.array([0,0,focus+0.05])
# # 
# # print(focal_length)
# # 
# beam1 = ray.Beam(20, 5e-3)
# # 
# # 
# # create spherical lens and output plane
# spher_lens = ray.SphericalRefraction(spher_location, curv, ref_1, ref_2, 1e-2)


#print(spher_lens.spot_plot(beam1))

# rms_size = np.divide(
#     np.sum(
#         np.square(
#             np.linalg.norm(beam1.vertices()[:,-1,:2], 
#                            axis=1)
#             )
#         ), 
#     beam1.size())

# rms_size = np.divide(np.sqrt(np.sum(np.square(np.linalg.norm(beam1.vertices()[:,-1,:2], axis=1)))), beam1.size())

# print(rms_size)

# focus_estimate = ray.focal_point([spher_lens])

# error_focal_estimate = np.abs(focus_estimate - focus) / focus

# print(error_focal_estimate)

# # 
# # print(spher_lens.spot_plot(beam1))
# # 
# plt.figure()
# beam1.show2D()

# out = ray.OutputPlane(np.array([0,0,focus_estimate]), x_len=1)
# # 
# spher_lens.propagate_beam(beam1)
# out.propagate_beam(beam1)
# # 
# plt.figure()
# beam1.show2D()
# #print(beam1.vertices()[:,:,2])
# spher_lens.show2D()
# 
# plt.figure()
# plt.scatter(beam1.vertices()[:,0,0], beam1.vertices()[:,0,1])

#print(beam1.vertices()[i,-1,0])


# =============================================================================
# Led to creation of a spot_plot method in the SphericalRefraction class 
# which calculated the theoretical location of the focus using a paraxial
# approximation, and then plots it as well as returning the rms spot size.
# Focal lengths appear to be correct as almost identical to theoretical value.
# =============================================================================


#%%
# =============================================================================
# Test 6 - Modelling a Plano-Convex Lens
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
#out = ray.OutputPlane(np.array([0,0,0.15]), x_len=1, y_len=1)

#plano_side1.propagate_beam(beam1)
#convex_side1.propagate_beam(beam1)
#out.propagate_beam(beam1)


system1 = [plano_side1, convex_side1]

plt.figure()
rms_size1 = ray.spot_plot(beam1, system1)

print(rms_size1)

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

# # create plano-convex singlet
plano_side2 = ray.SphericalRefraction(plane_location, plane_curv, plane_ref_1, plane_ref_2, 1e-2)
convex_side2 = ray.SphericalRefraction(con_location, con_curv, con_ref_1, con_ref_2, 1e-2)
out2 = ray.OutputPlane(np.array([0,0,0.15]), x_len=1, y_len=1)

system2 = [convex_side2, plano_side2]

convex_side2.propagate_beam(beam2)
plano_side2.propagate_beam(beam2)
out2.propagate_beam(beam2)

# rms_size2 = ray.spot_plot(beam2, system2)

# print(rms_size2)

# plt.figure()
# plt.scatter(beam1.vertices()[:,0,0], beam1.vertices()[:,0,1])

# plt.figure()
# plt.scatter(beam2.vertices()[:,0,0], beam2.vertices()[:,0,1])

plt.figure()
beam1.show2D(style='line')
plano_side1.show2D()
convex_side1.show2D()

plt.figure()
beam2.show2D()
plano_side2.show2D()
convex_side2.show2D()

plt.figure()

ax1 = plt.axes(projection='3d')
beam1.show3D(ax1, axes_order='zyx')

plt.figure()

ax2 = plt.axes(projection='3d')
beam2.show3D(ax2, axes_order='zyx')


# =============================================================================
# Initial issues with the plane first convex second configuration. Rays seemed
# to diverge at the final surface despite going from a higher refractive index
# to a lower one.
# Used a different variation of a 3D vector generalisation of snells law which
# gave the correct shapes of beams i.e. converge/diverge when they should.
# However, the calculated focus was closer for the convex->plane configuration
# than it should be.
# Upon inspecting the ray directions, they had a negative z component despite
# propagating to each surface. This led to a condition in the intercept method
# being imposed to check for this, but it strangely had no effect. My feeling
# is that there is a double negative multiplication somewhere in the intercept
# or snell methods which is causing this.
# My feeling is I'm defining the surface intercept vector incorrectly, but I 
# couldn't figure out the exact root cause of the issue.
# I decided to continue with the next task and come back to this issue if there
# was time, as lots of time had already been spent trying to fix this issue.
# =============================================================================


#%%
# =============================================================================
# Test 7 - Plano-Convex Lens Optimisation
# =============================================================================

curvatures = [20, 0]

# out = ray.plano_convex_optimise([100e-3, 5e-3, 1, 1.5168, 1e-2], curvatures)
# print()
# print(out)

'''
Below is further testing to try to figure out why the above line of code 
doesn't give desired output.
'''

# plane side information
plane_location = 105e-3
plane_curv = curvatures[1]
plane_ref_1 = 1.5168
plane_ref_2 = 1

# convex side information

con_location = plane_location - 5e-3
con_curv = curvatures[0]
con_ref_1 = 1
con_ref_2 = 1.5168

beam2 = ray.Beam(30, 5e-3)
#print(beam2.__str__())

# # create plano-convex singlet
plano_side2 = ray.SphericalRefraction(plane_location, plane_curv, plane_ref_1, plane_ref_2, 1e-2)
convex_side2 = ray.SphericalRefraction(con_location, con_curv, con_ref_1, con_ref_2, 1e-2)
out2 = ray.OutputPlane(np.array([0,0,0.15]), x_len=1, y_len=1)
print(out2.__str__())
system2 = [convex_side2, plano_side2]

rms_size2 = ray.spot_plot(beam2, system2)

convex_side2.propagate_beam(beam2)
plano_side2.propagate_beam(beam2)
out2.propagate_beam(beam2)

plt.figure()
beam2.show2D(style='line')

#print(beam1.vertices()[1,-1,2] - beam2.vertices()[1,-2,2])

# print(rms_size2)

focal_length = 1/(curvatures[0]*(con_ref_2-1))
print('Theoretical Focal Length = ' + str(focal_length))
focus = focal_length + con_location# + (1/curvatures[0])
print('Theoretical Focus z Location = ' + str(focus))

focus_estimate = ray.focal_point([convex_side2])
# print(focus_estimate)
print('Numerical Estimate of Focus Location = ' + str(focus_estimate))
# focus = 1/(curvatures[0]*(con_ref_2-1))

error_focal_estimate = np.abs(focus_estimate - focus)
print('Deviation in Focus Locations = ' + str(error_focal_estimate))

# =============================================================================
# Have reviewed code in detail and can't seem to get the plano_convex_optimise
# function working as it should. It works for some initial guesses but not all.
# Essentially, the function reviews a few curvatures around the initial guess,
# but not many at all, and then returns the initial guesses rather than new 
# curvatures. Looking into the process in detail, it seems as though rays are
# not reaching the second surface for curvatures that should allow them to.
# I added a means of creating new beam and lens objects with different names
# after each time the rms_size method is called to avoid memory issues.
# The result of the minimisation also seems very sensitive to the initial
# guesses, only testing curvatures very close to them (~ within less than 0.1%).
# After thoroughly reviewing the code, I see no resdon why it shouldn't work,
# so I have a feeling its due to the previous bug outlined in the prior test
# section (see above).
#
# Edit: Seems like it might actually be an issue with scipy.optimize.minimize()
# as this may not work properly for two variable functions, at least not in my
# use case.
# =============================================================================


#%%
# =============================================================================
# Test 8 - Focus for a single spherical surface (retrying given using new
# equations/functionality)
# =============================================================================

surface_loc = 100e-3
curv = 20
ref1 = 1
ref2 = 1.5168

spher_surf = ray.SphericalRefraction(surface_loc, curv, ref1, ref2, 1e-2)
focal_len = ref2 / (curv * (ref2-ref1))
#focus_loc = surface_loc + focal_len
focus_computed = ray.focal_point([spher_surf]) - surface_loc

print('Theoretical Focal Length is %s'%(focal_len))
print('Computed Focal Length is %s'%(focus_computed))

surface_loc = 100e-3
curv = -20
ref1 = 1.5168
ref2 = 1

spher_surf1 = ray.SphericalRefraction(surface_loc, curv, ref1, ref2, 1e-2)
focal_len = ref2 / (curv * (ref2-ref1))
focus_computed = ray.focal_point([spher_surf1]) - surface_loc

print('Theoretical Focal Length is %s'%(focal_len))
print('Computed Focus Length is %s'%(focus_computed))

# =============================================================================
# Found bug for getting the surface normal for the snell's law calculation 
# which is now fixed. Was adding the position of the lens onto the surface 
# normal for some reason, which is why all rays were skewed by the same amount, 
# as well as why some curvatures looked fine and some didn't. Also now very 
# sure I'm using the correct 3D snell's law equation.
# The strange thing now however is that I'm still not getting correct focal
# length values. They are drastically different depending on the way the
# spherical surface is facing. Whilst I would expect a small difference, not 
# one as big as this.
# Have gone through the maths of the focal_point function and it seems correct.
# From the plots in the previous section it looks as though the rays are barely
# being refracted off an initial spherical surface with positive curvature, but
# are for a final surface of negative nurvature.
# It looks almost correct in the second case anyway.
# Seems like there's an issue with the snell's law method, because it's the new
# direction after the surface that seems to be the issue given that it isn't
# giving the correct focus location for a single refracting surface.
# =============================================================================

