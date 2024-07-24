#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:06:21 2021

@author: Ruadhri McDougall

This file is a library containing all the relevant classes and functions for
the raytracer python project.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import fmin


class Ray:
    '''
    A class that contains all the properties of a ray of light.
    
    This class contains methods to operate on that ray of light.
    '''
    
    def __init__(self, start_pos = np.zeros(3), start_dir = np.array([0,0,1])):
        '''

        Parameters
        ----------
        start_pos : TYPE, np.ndarray
            DESCRIPTION. The initial position of the ray. The default is 
                         np.array([0,0,0]).
        start_dir : TYPE, np.ndarray
            DESCRIPTION. The initial direction of the ray. The default is 
                         np.array([0,0,1]).

        Raises
        ------
        TypeError
            DESCRIPTION. error raised when input for start position or 
                         direction is not a numpy array
        ValueError
            DESCRIPTION. error raised when ray starting position or direction 
                         is not of size 3, i.e. of form [x, y, z]

        Class Variables
        -------------      
        _start_position : TYPE, np.ndarray
            DESCRIPTION. Array for start position.
        _start_direction : TYPE, np.ndarray
            DESCRIPTION. Array for start direction.
        _all_points : TYPE, np.ndarray
            DESCRIPTION. Contains all points along the ray, where it intersects
                         with objects and output.
        _all_directions : TYPE, np.ndarray
            DESCRIPTION. Contains a record of all of all previous ray 
            directions
        _terminated : TYPE, bool
            DESCRIPTION. Defines the termination status of the ray.
        
        '''
        if not isinstance(start_pos, np.ndarray):
            raise TypeError('Ray starting position must be a 1D numpy array of' 
                            + ' length 3')
        elif start_pos.size != 3:
            raise ValueError('Ray starting position must be a 1D numpy array' 
                             + ' of length 3')
        if not isinstance(start_dir, np.ndarray):
            raise TypeError('Ray starting direction must be a 1D numpy array' 
                            + ' of length 3')
        elif start_dir.size != 3:
            raise ValueError('Ray starting direction must be a 1D numpy array' 
                             + ' of length 3')
        # the astype(float) statement ensures all indeces are float variables   
        self._start_position = start_pos.astype(float)
        self._start_direction = start_dir.astype(float)
        self._all_points = np.zeros((1,3)) + self._start_position
        self._all_directions = np.zeros((1,3)) + self._start_direction
        self._terminated = False
    
    def p(self):
        '''

        Returns
        -------
        TYPE np.ndarray
            DESCRIPTION. returns the current ray position

        '''
        return self._all_points[-1,:].astype(float)
    
    def k(self):
        '''

        Returns
        -------
        TYPE np.ndarray
            DESCRIPTION. returns the current ray direction

        '''
        return self._all_directions[-1,:].astype(float)
    
    def append(self, new_point, new_dir):
        '''

        Parameters
        ----------
        new_point : TYPE np.ndarray
            DESCRIPTION. the new point to be added to the array of points along
                         the ray
        new_dir : TYPE np.ndarray
            DESCRIPTION. the new direction of the ray

        Raises
        ------
        TypeError
            DESCRIPTION. error raised when input for position or direction is 
                         not a numpy array
        ValueError
            DESCRIPTION. error raised when ray position or direction is not of 
                         size 3, i.e. of form [x, y, z]

        '''
        if not isinstance(new_point, np.ndarray):
            raise TypeError('New ray position must be a 1D numpy array of' 
                            + ' length 3')
        elif new_point.size != 3:
            raise ValueError('New ray position must be a 1D numpy array of' 
                             + ' length 3')
        if not isinstance(new_dir, np.ndarray):
            raise TypeError('New ray direction must be a 1D numpy array of' 
                            + ' length 3')
        elif new_dir.size != 3:
            raise ValueError('New ray direction must be a 1D numpy array of' 
                             + ' length 3')
        self._all_points = np.vstack((self._all_points, new_point))
        self._all_directions = np.vstack((self._all_directions, new_dir))
    
    def vertices(self):
        '''

        Returns
        -------
        TYPE np.ndarray
            DESCRIPTION. returns all points along the ray

        '''
        return self._all_points
    
    def all_ray_directions(self):
        '''

        Returns
        -------
        TYPE np.ndarray
            DESCRIPTION. returns an array containing all previous directions
                         of the ray.

        '''
        return self._all_directions
    
    def copy(self):
        # create a new beam with the same parameters
        pass
    
    def terminate(self, status):
        '''

        Parameters
        ----------
        status : TYPE bool
            DESCRIPTION. Sets the status of the ray; True for terminated and
                         False for propagating

        Raises
        ------
        TypeError
            DESCRIPTION. Raised if input is not boolean.

        '''
        if status == True:
            self._terminated = True
        elif status == False:
            self._terminated = False
        else:
            raise TypeError('Input to terminate method must be boolean.')
            
    def termination(self):
        '''
        
        Raises
        ------
        TypeError
            DESCRIPTION. Raised if input is not boolean


        Returns
        -------
        bool
            DESCRIPTION. Returns the termination status of the ray; True for 
                         terminated and False for propagating.
                         
                         This is used in propagate_ray methods as terminated
                         rays are no longer allowed to propagate.

        '''
        if self._terminated == True:
            return True
        elif self._terminated == False:
            return False
        else:
            raise TypeError('Input to termination method must be boolean.')
    
    def show2D(self, plot_axes='zy', ray_colour='red'):
        '''

        Plots the ray path in 2D using matplotlib.
        
        The user can create their own figure on which to plot multiple 
        rays/beams, and to create a title / axis labes use the relevant 
        matplotlib commands after calling this method.
        
        Parameters
        ----------
        plot_axes : TYPE, optional
            DESCRIPTION. The default is 'zy', where z is the horizontal axis
                         and y is the vertical axis.
                         The options are 'zy', 'zy' or 'xz'.
        ray_colour : TYPE, optional
            DESCRIPTION. The default is 'red'.

        Raises
        ------
        ValueError
            DESCRIPTION. Error raised if something than the above stated
                         options are used for the plot_axes input.

        '''
        ray_positions = self.vertices()
        if plot_axes == 'zy':
            horiz_vals = ray_positions[:,2]
            vert_vals = ray_positions[:,1]
        elif plot_axes == 'xy':
            horiz_vals = ray_positions[:,0]
            vert_vals = ray_positions[:,1]
        elif plot_axes == 'xz':
            horiz_vals = ray_positions[:,0]
            vert_vals = ray_positions[:,2]
        else:
            raise ValueError('Input for plot_axes must be xy, xz or zy. The '
                             + 'default input is zy.')
        plt.plot(horiz_vals, vert_vals, color=ray_colour)
    
    def show3D(self, axes, axes_order='zyx', ray_colour='red'):
        '''
        Plots the ray on a set of 3D axes.
        
        The user can create their own figure on which to plot multiple 
        rays/beams, and to create a title / axis labes use the relevant 
        matplotlib commands after calling this method.
        
        To do this, it requires the user to call the command,
        
                       axes = plt.axes(projection='3d')
        
        where ax is the plotting axes used as an input to this method.

        Parameters
        ----------
        ax : TYPE
            DESCRIPTION. The plotting axes used (see above).
        axes_order : TYPE, optional
            DESCRIPTION. The default is 'zyx'. Used to specify the order in
                         which data is plotted to best orientate the graph.
                         e.g. z is horizontal for zyx.
        ray_colour : TYPE, optional
            DESCRIPTION. The default is 'red'.

        Raises
        ------
        ValueError
            DESCRIPTION. Error raised if something than the above stated
                         options are used for the axes_order input.

        '''
        ray_positions = self.vertices()
        if axes_order == 'zyx':
            axes.plot(ray_positions[:,2], 
                      ray_positions[:,1], 
                      ray_positions[:,0],
                      color=ray_colour)
        elif axes_order == 'xyz':
            axes.plot(ray_positions[:,0], 
                      ray_positions[:,2], 
                      ray_positions[:,1],
                      color=ray_colour)
        elif axes_order == 'xzy':
            axes.plot(ray_positions[:,0], 
                      ray_positions[:,1], 
                      ray_positions[:,2],
                      color=ray_colour)
        else:
            raise ValueError('Input for axes_order must be xyz, xzy or zyx. '
                             + 'The default input is zyx.')
    
    def __repr__(self):
        return '%s(pos=%s, dir=%s)'%('Ray', self.p(), self.k())
    
    def __str__(self):
        return ('%s - Current ray position is %s and current ray direction is %s'
                %('Ray Object', self.p(), self.k()))
    
    
class Beam:
    '''
    The Beam class represents a bundle of rays with a uniform density.
    '''
    
    def __init__(self, num_rays, radius, layers=5, arrange='circle', 
                 start_pos=np.zeros(3), start_dir=np.array([0,0,1])):
        '''
        
        Parameters
        ----------
        num_rays : TYPE int
            DESCRIPTION. In the case where arrange == 'circle', this is the
                         number of rays per radial layer within the beam.
                         In the case where arrange == 'square', this is the
                         number of rays within a square with side length equal
                         to the beam diameter (does not represent total num 
                         rays in the beam).
        radius : TYPE float
            DESCRIPTION. the beam radius
        layers : TYPE, optional
            DESCRIPTION. The number of layers of rays within the beam. The 
                         default is 5.
        arrange : TYPE, optional
            DESCRIPTION. Defines how the beam is setup, either with a linear 
                         square density within a circle ('square'), or with a 
                         uniform radial density ('circle'). The default is 
                         'circle'.
        start_pos : TYPE, optional
            DESCRIPTION. Where the beam begins, taken as the centre of the beam 
                         circle. The default is np.array([0,0,0]).
        start_dir : TYPE, optional
            DESCRIPTION. The initial direction of each ray in the beam, where 
                         the default is np.array([0,0,1]).

        Raises
        ------
        TypeError
            DESCRIPTION. Error raised when above paramaters aren't entered as
                         their specified variable types.
        ValueError
            DESCRIPTION. Error when start_pos or start_dir are not numpy arrays
                         of length 3, i.e. np.array([i, j, k])

        Class Variables
        -------------      
        _all_rays : TYPE list
            DESCRIPTION. a list of all the Ray objects that make up the beam
        _radius : TYPE float
            DESCRIPTION. The beam radius
        _start_position : TYPE np.ndarray
            DESCRIPTION. The start position of the beam centre
        _start_direction : TYPE np.ndarray
            DESCRIPTION. The initial direction of each ray within the beam
        _ray_count : TYPE int
            DESCRIPTION. The number of rays that make up the beam.

        '''
        if not isinstance(start_pos, np.ndarray):
            raise TypeError('Beam starting position must be a 1D numpy array' 
                            + ' of length 3')
        elif start_pos.size != 3:
            raise ValueError('Beam starting position must be a 1D numpy array' 
                             + ' of length 3')
        if not isinstance(start_dir, np.ndarray):
            raise TypeError('Beam starting direction must be a 1D numpy array' 
                            + ' of length 3')
        elif start_dir.size != 3:
            raise ValueError('Beam starting direction must be a 1D numpy array' 
                             + ' of length 3')
        if not isinstance(num_rays, int):
            raise TypeError('Number of rays in beam must be an integer')
        if not isinstance(radius, float) and not isinstance(radius, int):
            raise TypeError('radius input must be a float or integer.')
        
        if arrange == 'circle':
            # num_rays defines the number of rays per layer within the beam
            thetas = np.linspace(0, 2*np.pi, num_rays)
            inner_radius = radius / layers
            ray_lst = []
            new_radius = inner_radius
            for l in range(layers):
                x_pos = new_radius * np.sin(thetas)
                y_pos = new_radius * np.cos(thetas)
                for t in range(num_rays):
                    ray_pos = (np.array([x_pos[t], y_pos[t], 0]) 
                               + start_pos.astype(float))
                    ray_lst.append(Ray(ray_pos, start_dir))
                new_radius += inner_radius
        
        elif arrange == 'square':
            rays_per_dir = int(np.rint(np.sqrt(num_rays)))
            ray_pos_x = np.linspace(-1*radius, radius, rays_per_dir)
            ray_pos_y = np.linspace(-1*radius, radius, rays_per_dir)
            # meshgrid allows magnitudes to be calculated only once
            all_xy = np.array(np.meshgrid(ray_pos_x, ray_pos_y))
            ray_position_mags = np.linalg.norm(all_xy, axis=0)
            ray_lst = []
            # have to loop through each start position to create each new ray
            # one at a time
            for x in range(rays_per_dir):
                for y in range(rays_per_dir):
                    #print([ray_pos_x[x], ray_pos_y[y]])
                    #print(ray_position_mags[x,y] <= radius)
                    if ray_position_mags[x,y] <= radius:
                        ray_position = np.array([ray_pos_x[x], 
                                                 ray_pos_y[y], 
                                                 0])
                        ray_position = ray_position + start_pos
                        ray_lst.append(Ray(ray_position, start_dir))
        else:
            raise ValueError('Arrange input must be a string of either '
                             + '"circle" or "square"')
        self._all_rays = ray_lst
        self._radius = radius
        self._start_position = start_pos.astype(float)
        self._start_direction = start_dir.astype(float)
        self._ray_count = len(ray_lst)
        self._config = arrange
        

    def p(self):
        '''

        Returns
        -------
        points_lst : TYPE list
            DESCRIPTION. returns the current points of each ray in the beam

        '''
        points_lst = []
        for i in range(self._ray_count):
            current_pos_i = self._all_rays[i].p()
            points_lst.append(current_pos_i)
            return points_lst

    def k(self):
        '''

        Returns
        -------
        dir_lst : TYPE list
            DESCRIPTION. returns the current directions of each ray in the beam

        '''
        dir_lst = []
        for i in range(self._ray_count):
            current_dir_i = self._all_rays[i].k()
            dir_lst.append(current_dir_i)
        return dir_lst
    
    def size(self):
        '''

        Returns
        -------
        TYPE list
            DESCRIPTION. returns the number of rays in the beam

        '''
        return self._ray_count
    
    def all_rays(self):
        '''

        Returns
        -------
        TYPE list
            DESCRIPTION. A list of all the ray objects that form the beam class 

        '''
        return self._all_rays
    
    def vertices(self):
        '''

        Returns
        -------
        beam_pts : TYPE np.ndarray
            DESCRIPTION. Returns all the points that each ray in the beam
                         propagates through.
                         
                         The output array is 3D and has dimensions,
                         (no. rays, points per ray, 3)

        '''
        points_per_ray = len(self._all_rays[0].vertices())
        beam_pts = np.zeros((self._ray_count, points_per_ray, 3))
        for i in range(self._ray_count):
            vertices_i = self._all_rays[i].vertices()
            beam_pts[i, :, :] += vertices_i
        return beam_pts
        
    def all_beam_directions(self):
        '''

        Returns
        -------
        directions_lst : TYPE list
            DESCRIPTION. Returns all the directions that the ray had during
                         its propagation.
                         
                         The output array in 3D and has dimensions
                         (no. rays, directions per ray, 3)

        '''
        dir_per_ray = len(self._all_rays[0].all_ray_directions())
        all_beam_dir = np.zeros((self._ray_count, dir_per_ray, 3))
        for i in range(self._ray_count):
            directions_i = self._all_rays[i].all_ray_directions()
            all_beam_dir[i, :, :] = directions_i
        return all_beam_dir
    
    def show2D(self, plot_axes='zy', beam_color='blue', style='line'):
        '''
        
        Plots the beam path in 2D using matplotlib.
        
        The user can create their own figure on which to plot multiple 
        rays/beams, and to create a title / axis labes use the relevant 
        matplotlib commands after calling this method.

        Parameters
        ----------
        plot_axes : TYPE, optional
            DESCRIPTION. The default is 'zy', where z is the horizontal axis
                         and y is the vertical axis.
                         The options are 'zy', 'zy' or 'xz'.
        beam_color : TYPE, optional
            DESCRIPTION. The default is color 'blue'.

        Raises
        ------
        ValueError
            DESCRIPTION. Error raised if something than the above stated
                         options are used for the plot_axes input.
                         
                         Error also raised if input for style is not 'line' or
                         'scatter'.

        '''
        if style == 'line':
            all_beam_points = self.vertices()
            if plot_axes == 'zy':
                for i in range(self._ray_count):
                    plt.plot(all_beam_points[i,:,2], 
                             all_beam_points[i,:,1], 
                             color=beam_color)
            elif plot_axes == 'xy':
                for i in range(self._ray_count):
                    plt.plot(all_beam_points[i,:,0], 
                             all_beam_points[i,:,1], 
                             color=beam_color)
            elif plot_axes == 'xz':
                for i in range(self._ray_count):
                    plt.plot(all_beam_points[i,:,0], 
                             all_beam_points[i,:,2], 
                             color=beam_color)
            else:
                raise ValueError('Input for plot_axes must be xy, xz or zy. '
                                 + 'The default input is zy.')
        elif style == 'scatter':
            all_beam_points = self.vertices()
            if plot_axes == 'zy':
                for i in range(self._ray_count):
                    plt.scatter(all_beam_points[i,:,2], 
                                all_beam_points[i,:,1], 
                                color=beam_color)
            elif plot_axes == 'xy':
                for i in range(self._ray_count):
                    plt.scatter(all_beam_points[i,:,0], 
                                all_beam_points[i,:,1], 
                                color=beam_color)
            elif plot_axes == 'xz':
                for i in range(self._ray_count):
                    plt.scatter(all_beam_points[i,:,0], 
                                all_beam_points[i,:,2], 
                                color=beam_color)
            else:
                raise ValueError('Input for plot_axes must be xy, xz or zy. '
                                 + 'The default input is zy.')
        else:
            raise ValueError('Input for plot style "line" or "scatter".')
        
    def show3D(self, axes, axes_order='zyx', beam_color='blue'):
        '''
        Plots the beam on a set of 3D axes.
        
        The user can create their own figure on which to plot multiple 
        rays/beams and spherical lens objects, and to create a title / axis 
        labels use the relevant matplotlib commands after calling this method.
        
        To do this, it requires the user to call the command,
        
                         ax = plt.axes(projection='3d')
        
        where ax is the plotting axes used as an input to this method.
        
        Parameters
        ----------
        ax : TYPE object
            DESCRIPTION. The plotting axes used (see above).
        axes_order : TYPE, optional
            DESCRIPTION. The default is 'zyx'. Used to specify the order in
                         which data is plotted to best orientate the graph.
                         e.g. z is horizontal for zyx.
        beam_color : TYPE, optional
            DESCRIPTION. The default is 'blue'.

        Raises
        ------
        ValueError
            DESCRIPTION. Error raised if something than the above stated
                         options are used for the axes_order input.

        '''
        all_beam_points = self.vertices()
        if axes_order == 'zyx':
            for i in range(self._ray_count):
                axes.plot(all_beam_points[i,:,2], 
                          all_beam_points[i,:,1],
                          all_beam_points[i,:,0], 
                          color=beam_color)
        elif axes_order == 'xyz':
            for i in range(self._ray_count):
                axes.plot(all_beam_points[i,:,0], 
                          all_beam_points[i,:,1],
                          all_beam_points[i,:,2], 
                          color=beam_color)
        elif axes_order == 'xzy':
            for i in range(self._ray_count):
                axes.plot(all_beam_points[i,:,0], 
                          all_beam_points[i,:,2],
                          all_beam_points[i,:,1], 
                          color=beam_color)
        else:
            raise ValueError('Input for plot_axes must be xy, xz or yz. The '
                             + 'default input is yz.')

    def __repr__(self):
        return ('%s(num=%s, r=%s, config=%s)'%('Beam', self._ray_count, 
                                               self._radius, self._config)
                )
    
    def __str__(self):
        return ('%s - Contains %s rays, has a radius of %sm and a %s configuration'
                %('Beam Object', self._ray_count, self._radius, self._config))
    

class Pointsource:
    '''
    As an extension, I would have liked to add functionality to create a point
    source. 
    
    This would be fairly trivial and be set up similar to the beam
    class but with all rays having the same start position and having
    directions specified by an equal steradian spacing.
    
    It would be useful for modelling optical fibre output for example.
    '''
    pass


class OpticalElement:
    '''
    
    A base class for all optical elements.
    
    Not really necessary to have this and so wasn't used.
    
    '''
    def propagate_ray(self, ray):
        "propagate a ray through the optical element"
        raise NotImplementedError()


class SphericalRefraction:
    """
    Spherical lens object
    
    """
    
    def __init__(self, z_0, curvature, n_1, n_2, aperture_radius):
        '''

        Parameters
        ----------
        z_0 : TYPE float
            DESCRIPTION. A float that defines the position of the intersection 
                         of the initial surface of the object with the z axis
        curvature : TYPE float
            DESCRIPTION. The curvature of the lens (1 / radius of curvature)
        n_1 : TYPE float
            DESCRIPTION. The refractive index of the initial medium
        n_2 : TYPE float
            DESCRIPTION. The refractive index of the refracted medium
        aperture_radius : TYPE float
            DESCRIPTION. The radius of the aperture the ray is allowed to
                         propagate through

        Raises
        ------
        TypeError
            DESCRIPTION. Error raised if above inputs are not of specified 
                         types
        ValueError
            DESCRIPTION. Error if aperture radius is greater than the radius of 
                         curvature, since this does not make physical sense

        Class Variables
        ---------------      
        _surface_intercept : TYPE, np.ndarray
            DESCRIPTION. Defines the intersection point of the object with the 
                         z-axis
        _curvature : TYPE, float
            DESCRIPTION. Array for start direction
        _ref_index_1 : TYPE, float
            DESCRIPTION. The refractive index of the initial medium
        _ref_index_2 : TYPE, float
            DESCRIPTION. The refractive index of the refracted medium
        _aperture_radius : TYPE, float
            DESCRIPTION. The radius of the aperture the ray is allowed to
                         propagate through
        _radius_curvature : TYPE, float
            DESCRIPTION. The radius of curvature of the lens
        _surface_normal : TYPE, np.ndarray
            DESCRIPTION. The normalised surface normal vector at the point the
                         ray intersects with the lens. This changes for a given
                         ray each time the propagate_ray method is called.

        '''
        
        if not isinstance(z_0, float) and not isinstance(z_0, int):
            raise TypeError('Object intercept position must be a float or an '
                            + 'integer, indicating the initial surface '
                            + 'position along the z axis')
        if not isinstance(curvature, float) and not isinstance(curvature, int):
            raise TypeError('Curvature must be a float or an integer')
        if not isinstance(n_1, float) and not isinstance(n_1, int):
            raise TypeError('Refractive indeces must be a float or integer')
        if not isinstance(n_2, float) and not isinstance(n_2, int):
            raise TypeError('Refractive indeces must be a float or integer')
        if (not isinstance(aperture_radius, float) 
            and not isinstance(aperture_radius, int)):
            raise TypeError('Refractive indeces must be a float or integer')
        # the astype(float) statement ensures all indeces are float variables
        self._surface_intercept = np.array([0, 0, z_0]).astype(float)
        self._curvature = curvature
        self._ref_index_1 = n_1
        self._ref_index_2 = n_2
        self._aperture_radius = aperture_radius
        if curvature != 0:
            self._radius_curvature = 1 / self._curvature
        else:
            self._radius_curvature = None
        if self._radius_curvature is not None:
            if aperture_radius > np.abs(self._radius_curvature):
                raise ValueError('An aperture radius greater than the radius '
                                 + 'of curvature does not make physical sense')
    
    def intercept(self, ray):
        '''
        
        Uses geometry and vector relationships to calculate the cartesian 
        location of the intersection of a line vector with a sphere.
        
        The geometric equation used generates two solutions, where the correct
        solution is determined dependent on the radius of curvature.
        
        There is an edge case when the curvature is zero, corresponding to a 
        vertical surface, and so this is considered separately.
        
        Parameters
        ----------
        ray : TYPE Object
            DESCRIPTION. ray is an instance of the Ray class
        
        Raises
        ------
        TypeError
            DESCRIPTION. When input is not a Ray object

        Returns
        -------
        ray_intersect_loc : TYPE np.ndarray
            DESCRIPTION. a numpy array of size 3 which describes the location
                         of the intersection between the ray and the object.
                         
                         If the location of intersection is outside the given
                         aperture radius, the value of None is returned.
        
        '''
        if not isinstance(ray, Ray):
            raise TypeError('Input to intercept method must be a Ray '
                            + 'object')
        ray_pos = ray.p()
        norm_ray_dir = ray.k() / np.linalg.norm(ray.k())
        object_centre = np.zeros(3) + self._surface_intercept
        if self._curvature != 0:
            # this assignment is just to allow the 2nd index to be altered
            object_centre[2] = object_centre[2] + self._radius_curvature
            self._centre_pt = object_centre
            object_cent_to_ray = ray_pos - object_centre
            object_cent_to_ray_mag = np.linalg.norm(object_cent_to_ray)
            dot_prod_ray_object_pos = np.dot(object_cent_to_ray, norm_ray_dir)
            magnitude = (dot_prod_ray_object_pos**2 
                         - object_cent_to_ray_mag**2 
                         + self._radius_curvature**2)
            if self._radius_curvature > 0:
                ray_to_obj_len = (-1 * dot_prod_ray_object_pos 
                                  - np.sqrt(magnitude)
                                  )
            elif self._radius_curvature < 0:
                ray_to_obj_len = (-1 * dot_prod_ray_object_pos 
                                  + np.sqrt(magnitude)
                                  )
            ray_to_obj_edge = norm_ray_dir * ray_to_obj_len
            ray_intersect_loc = ray_pos + ray_to_obj_edge
        elif self._curvature == 0:
            ray_to_obj_len = -1 * np.dot((ray_pos - self._surface_intercept),
                                         norm_ray_dir)
            ray_to_obj_edge = norm_ray_dir * ray_to_obj_len
            ray_intersect_loc = ray_pos + ray_to_obj_edge
            magnitude = 1
        aperture_area = ray_intersect_loc[0]**2 + ray_intersect_loc[1]**2
        if aperture_area < self._aperture_radius**2 and magnitude > 0:
            return ray_intersect_loc
        else:
            return None
    
    def snell(self, ray):
        '''
        Uses a 3D vector generalisation of snells law to calculate the new
        direction of a ray when it intersects with a spherical refractive
        object.
        
        Parameters
        ----------
        incident_dir : TYPE np.ndarray
            DESCRIPTION. the incident direction of the ray
        surface_norm : TYPE np.ndarray
            DESCRIPTION. the vector normal to the refractive surface, with
                         direction defined in the direction of propagation
        n_index_1 : TYPE int /float
            DESCRIPTION. the refractive index of the first material
        n_index_2 : TYPE int / float
            DESCRIPTION. the refractive index of the second material
    
        Raises
        ------
        TypeError
            DESCRIPTION. Raised when the above inputs are not of their
                         specified types.
        ValueError
            DESCRIPTION. Error when incident direction or surface normal numpy 
                         arrays are not of size 3 (e.g. [x,y,z])
    
        Returns
        -------
        output_dir : TYPE np.ndarray
            DESCRIPTION. a size 3 numpy array that represents the normalised 
                         direction of the outgoing ray
    
        '''
        
        if not isinstance(ray, Ray):
            raise TypeError('Input to snell method must be a Ray object.')
            
        if self._curvature == 0:
            surface_norm_unit = np.array([0,0,1]).astype(float)
        elif self._curvature < 0:
            surface_norm = ray.p() - self._centre_pt
            surface_norm_unit = np.divide(surface_norm, 
                                          np.linalg.norm(surface_norm))
        elif self._curvature > 0:
            surface_norm = self._centre_pt - ray.p()
            surface_norm_unit = np.divide(surface_norm, 
                                          np.linalg.norm(surface_norm))
        incident_dir = ray.k()
        incident_dir_unit = np.divide(incident_dir, 
                                      np.linalg.norm(incident_dir))
        mu_factor = self._ref_index_1 / self._ref_index_2
        theta_in = np.arccos(np.dot(incident_dir_unit, 
                                    surface_norm_unit)
                             )
        sin_theta_in = np.sin(theta_in)
        mu_recip = 1 / mu_factor
        if sin_theta_in == 0:
            return incident_dir
        elif sin_theta_in < mu_recip:
            output_dir = (np.sqrt(1 
                                  - mu_factor**2 
                                  * (1 - (np.dot(surface_norm_unit, 
                                                  incident_dir_unit))**2)
                                  )
                          * surface_norm_unit
                          + mu_factor * (incident_dir_unit 
                                          - np.dot(surface_norm_unit, 
                                                  incident_dir_unit) 
                                          * surface_norm_unit))
            
            output_dir[:2] = (output_dir[:2] 
                              * (self._ref_index_2 
                                  / np.abs((self._ref_index_2 
                                            - self._ref_index_1))))
            return output_dir
        else:
            return None
        
    def propagate_ray(self, ray):
        '''
        
        Propagates the ray through the lens and appends the new location and 
        direction of the ray to the Ray class.
        
        If the intersection point or the new direction is returned as None then
        the ray is terminated.
        
        If the ray is already terminated then None is returned.
        
        Parameters
        ----------
        ray : TYPE Object
            DESCRIPTION. The ray object propagating through the lens
        
        Raises
        ------
        TypeError
            DESCRIPTION. When input is not a Ray object

        '''
        if not isinstance(ray, Ray):
            raise TypeError('Input to propagate_ray method must be a Ray '
                            + 'object')
        
        if ray.termination() == False:
            object_intersect = self.intercept(ray)
            new_ray_direction = self.snell(ray)
            if object_intersect is None:
                print('Ray does not intersect with the spherical object and '
                      + 'therefore has been terminated. \n')
                ray.terminate(True)
                ray.append(ray.p(), ray.k())
            elif new_ray_direction is None:
                print('The ray has undergone total internal reflection and '
                      + 'therefore has been terminated. \n')
                ray.terminate(True)
                ray.append(ray.p(), ray.k())
            else:
                ray.append(object_intersect, new_ray_direction)
        else:
            ray.append(ray.p(), ray.k())
        
    def propagate_beam(self, beam):
        '''
        Propagates the beam through the lens and appends the new location and 
        direction of each ray to the Ray class, and all the information is then
        contained in the beam class.
        
        This is achieved by looping over each ray element that comprises the 
        beam.

        Parameters
        ----------
        beam : TYPE Object
            DESCRIPTION. An instance of the Beam class

        Raises
        ------
        TypeError
            DESCRIPTION. When input is not a Beam object

        Returns
        -------
        None.

        '''
        if not isinstance(beam, Beam):
            raise TypeError('Input to propagate_beam method must be a Ray '
                            + 'object')
        rays_of_beam = beam.all_rays()
        for i in range(beam.size()):
            self.propagate_ray(rays_of_beam[i])
        
        
    def show2D(self, num_points=50):
        '''
        Plots the spherical lens object on a set of 2D axes using matplotlib.
        This is useful to better visualise beam paths to understand where in 
        space their paths are altered.
        
        The user can create their own figure on which to plot multiple 
        rays/beams and lens objects, and to create a title / axis labels use
        the relevant matplotlib commands after calling this method.

        Parameters
        ----------
        num_points : TYPE int (optional)
            DESCRIPTION. The number of coordinates that show the lens on the
                         diagram. The default is 50.

        '''
        if not isinstance(num_points, int):
            raise TypeError('num_points input must be a an integer')
        y_vals = np.linspace(-1*self._aperture_radius, 
                             self._aperture_radius, 
                             num_points)
        if self._curvature > 0:
            z_vals = (-1*np.sqrt(self._radius_curvature**2 - y_vals**2) 
                      + self._surface_intercept[2]
                      + self._radius_curvature)
        elif self._curvature < 0:
            z_vals = (np.sqrt(self._radius_curvature**2 - y_vals**2) 
                      + self._surface_intercept[2]
                      + self._radius_curvature)
        else:
            z_vals = np.zeros(num_points) + self._surface_intercept[2]
        plt.plot(z_vals, y_vals, color='black')
    
    def show3D(self):
        '''
        A possible extension would have been to create a 3D wireframe plot of
        the spherical lens for more intuitive output graphics.
        '''
        pass
    
    def __repr__(self):
        return ('%s(z_0=%s, curv=%s, n_1=%s, n_2=%s, r_ap=%s)'
                %('SphericalRefraction', self._surface_intercept[2], 
                  self._curvature, self._ref_index_1, 
                  self._ref_index_2, self._aperture_radius))
    
    def __str__(self):
        return ('%s - Z axis interpect at z=%sm, curvature of %s, initial refractive index %s, final refractive index %s, aperture radius %sm'
                %('SphericalRefraction Object', self._surface_intercept[2], 
                  self._curvature, self._ref_index_1, 
                  self._ref_index_2, self._aperture_radius))


class OutputPlane:
    '''
    A class for a plane which defines the output location of the rays.
    '''
    def __init__(self, z_0, x_len=1.0, y_len=1.0, 
                 surface_norm=np.array([0,0,1])):
        '''

        Parameters
        ----------
        z_0 : TYPE  np.ndarray
            DESCRIPTION. A 1D array of size 3 that defines the position of the
                         intersection of the object with the z axis
        x_len : TYPE float
            DESCRIPTION. Defines the width (side to side) of the output plane
                         in the x direction. Can be used to take a slice
                         through the centre of a beam by being set to zero.
        y_len : TYPE
            DESCRIPTION. Defines the width (side to side) of the output plane
                         in the y direction
        surface_norm : TYPE, optional
            DESCRIPTION. The surface normal of the plane, defined in the 
                         direction of beam propagation. 
                         The default is np.array([0,0,1]).

        Raises
        ------
        TypeError
            DESCRIPTION. Raised if above inputs are not of their specified type
        ValueError
            DESCRIPTION. Error if z_0 is not a 1D array of size 3
        
        Class Variables
        -------------      
        _surface_intercept : TYPE, np.ndarray
            DESCRIPTION. Defines the intersection point of the output plane 
                         with the z-axis
        _x_length : TYPE, float
            DESCRIPTION. Length of output plane in x direction
        _y_length : TYPE, float
            DESCRIPTION. Length of output plane in y direction
        _surface_normal : TYPE, np.ndarray
            DESCRIPTION. The surface normal of the plane, defined in the 
                         direction of beam propagation. 

        '''
        if not isinstance(z_0, np.ndarray):
            raise TypeError('Object intercept position must be a 1D numpy' 
                            + ' array of length 3')
        elif z_0.size != 3:
            raise TypeError('Object intercept position must be a 1D numpy' 
                            + ' array of length 3')
        if not isinstance(surface_norm, np.ndarray):
            raise ValueError('surface_norm must be a 1D numpy array of length '
                             + '3')
        elif surface_norm.size != 3:
            raise ValueError('surface_norm must be a 1D numpy array of length '
                             + '3')
        if not isinstance(x_len, float) and not isinstance(x_len, int):
            raise TypeError('Refractive indeces must be a float or integer')
        if not isinstance(y_len, float) and not isinstance(y_len, int):
            raise TypeError('Refractive indeces must be a float or integer')
        self._surface_intercept = z_0.astype(float)
        self._x_length = float(x_len)
        self._y_length = float(y_len)
        self._surface_normal = surface_norm.astype(float)

    def intercept(self, ray):
        '''
        
        Calculates the cartesian location of the intercept of the ray and the 
        output plane.
        
        Parameters
        ----------
        ray : TYPE Object
            DESCRIPTION. A Ray class object being propagated through the system
        
        Raises
        ------
        TypeError
            DESCRIPTION. Raised if input is not a Ray object
            
        Returns
        -------
        intersect_loc : TYPE np.ndarray
            DESCRIPTION. The location where the ray meets the output plane

        '''
        if not isinstance(ray, Ray):
            raise TypeError('Input to intercept method must be a Ray object')
        ray_pos = ray.p()
        norm_ray_dir = ray.k() / np.linalg.norm(ray.k())
        ray_to_centre = self._surface_intercept - ray_pos
        screen_distance = np.divide(np.dot(ray_to_centre, 
                                           self._surface_normal),
                                    np.dot(norm_ray_dir, 
                                           self._surface_normal))
        intersect_loc = ray_pos + screen_distance * norm_ray_dir
        x_deviation = np.abs(intersect_loc[0])
        y_deviation = np.abs(intersect_loc[1])
        if self._x_length < x_deviation or self._y_length < y_deviation:
            return None
        else:
            return intersect_loc
        
    def propagate_ray(self, ray):
        '''
        Propagates the ray to the output plane, and appends the output plane
        intersection location and the current direction to the Ray object.
        
        If the ray does not intersect with the output plane it becomes
        terminated.
        
        If the ray is already terminated then None is returned.
        
        Parameters
        ----------
        ray : TYPE Object
            DESCRIPTION. A Ray object being propagated to the output plane.
        
        Raises
        ------
        TypeError
            DESCRIPTION. Raised if input is not a Ray object
        
        '''
        if not isinstance(ray, Ray):
            raise TypeError('Input to propagate_ray method must be a Ray '
                            + 'object')
        if ray.termination() == False:
            plane_intercept = self.intercept(ray)
            if plane_intercept is None:
                print('Ray did not reach the output plane and so has been '
                      + 'terminated. \n')
                ray.terminate(True)
                ray.append(ray.p(), ray.k())
            else:
                ray.append(plane_intercept, ray.k())
        else:
            ray.append(ray.p(), ray.k())
        
    def propagate_beam(self, beam):
        '''
        Propagates a beam object to the output plane by iterating over the
        propagate ray method for each ray object contained by a beam object.

        Parameters
        ----------
        beam : TYPE Object
            DESCRIPTION. The beam object being propagated to the output plane.

        Raises
        ------
        TypeError
            DESCRIPTION. Raised if input is not a Beam object.

        '''
        if not isinstance(beam, Beam):
            raise TypeError('Input to propagate_beam method must be a Beam '
                            + 'object')
        rays_of_beam = beam.all_rays()
        for i in range(beam.size()):
            self.propagate_ray(rays_of_beam[i])
    
    def __repr__(self):
        return ('%s(z_0=%s, x_len=%s, y_len=%s, norm=%s)'
                %('OutputPlane', self._surface_intercept[2], 
                  self._x_length, self._y_length, 
                  self._surface_normal))
    
    def __str__(self):
        return ('%s - Z axis interpect at z=%sm, length in x direction %sm, length in y direction %sm, surface normal vector %s)'
                %('OutputPlane Object', self._surface_intercept[2], 
                  self._x_length, self._y_length, 
                  self._surface_normal))
        
# =============================================================================
# Below are a series of generic functions which are useful tools for the 
# worksheet tasks.   
# =============================================================================

def focal_point(optical_system, z_0=1.0, test_ray_dev=1e-6):
    '''
    Parameters
    ----------
    optical_system : TYPE list
        DESCRIPTION. A list where each instance is an object for an element in 
                     the optical system, in order of occurence.
                     *Note that this should not include an output plane object.
                     
    z_0 : TYPE float (optional)
        DESCRIPTION. Arbitrary location of the distant output plane. 
                     The default is 1.0. If the focus of the optical system is 
                     likely to be greater than 1m then increase this quantity.
                     
    test_ray_dev : TYPE float (optional)
        DESCRIPTION. The deviation of the test ray from the z-axis. The test 
                     ray should be arbitrary close to the z-axis without being 
                     along it. The default is 1e-6.

    Returns
    -------
    focus : TYPE float
        DESCRIPTION. The z-axis location of the focus after the final optical 
                     element in the system.
    '''
    if not isinstance(optical_system, list):
        raise TypeError('Optical system must be a list of SphericalRefraction '
                        + 'objects.')
    if not isinstance(z_0, float) and not isinstance(z_0, int):
        raise TypeError('Arbitrary location of output plane z_0 must be a '
                        + 'float or integer.')
    if not isinstance(test_ray_dev, float):
        raise TypeError('Deviation of test ray from z axis must be a float')
    if isinstance(optical_system[-1], OutputPlane):
        raise TypeError('Please do not include an output plane in the optical ' 
                        + 'system list')
    # define output loc as some point arbitrarily far after the last optical
    # element
    test_ray = Ray(np.array([0,test_ray_dev,0]), np.array([0,0,1]))
    output = OutputPlane(np.array([0,0,z_0]), 
                         x_len=np.inf, 
                         y_len=np.inf)
    for i in range(len(optical_system)):
        optical_system[i].propagate_ray(test_ray)
        if test_ray.termination() == True:
            return None
    output.propagate_ray(test_ray)
    test_vertices = test_ray.vertices()
    final_ray_gradient = ((test_vertices[-1,1] - test_vertices[-2,1]) 
                          / (test_vertices[-1,2] - test_vertices[-2,2]))
    y_intercept = (test_vertices[-2,1] 
                   - final_ray_gradient 
                   * test_vertices[-2,2])
    focus = -1 * y_intercept / final_ray_gradient
    return focus

def get_rms_size(beam, optical_system, z_0=1.0, test_ray_dev=1e-6):
    '''
    Calculates RMS beam radius at the location of the focus at the end of the
    optical system.
    
    Parameters
    ----------
    beam : TYPE Object
        DESCRIPTION. The Beam object for which the RMS size is calculated
    optical_system : TYPE list
        DESCRIPTION. A list of SphericalRefraction objects which comprise the
                     optical system to be analysed.
                     *Note, should not include an output plane.
    z_0 : TYPE float (optional)
        DESCRIPTION. Arbitrary location of the distant output plane. 
                     The default is 1.0. If the focus of the optical system is 
                     likely to be greater than 1m then increase this quantity.
                     
    test_ray_dev : TYPE float (optional)
        DESCRIPTION. The deviation of the test ray from the z-axis. The test 
                     ray should be arbitrary close to the z-axis without being 
                     along it. The default is 1e-6.

    Returns
    -------
    rms_rad : TYPE float
        DESCRIPTION. The RMS beam radius at the beam focus
    '''
    if not isinstance(beam, Beam):
        raise TypeError('Input to propagate_beam method must be a Beam '
                        + 'object')
    # No need for additional typechecks as all inputs go into the focal_point
    # method, where they will be checked there
    focus = focal_point(optical_system, z_0, test_ray_dev)
    output_plane = OutputPlane(np.array([0, 0, focus]))
    for i in range(len(optical_system)):
        optical_system[i].propagate_beam(beam)
    output_plane.propagate_beam(beam)
    ray_dist_to_axis = np.linalg.norm(beam.vertices()[:,-1,:2], axis=1)
    squares = np.square(ray_dist_to_axis)
    sum_squares = np.sum(squares)
    sum_squares_over_no_beams = np.divide(sum_squares, beam.size())
    rms_rad = np.sqrt(sum_squares_over_no_beams)
    return rms_rad

def spot_plot(beam, optical_system, z_0=1.0, test_ray_dev=1e-6, label=None):
    '''
    Takes a slice through the beam its focus though the z axis, and plots the 
    location of rays as a result, as well as showing the rms radius.
    
    Also returns the RMS radius at the focus.
    
    Parameters
    ----------
    beam : TYPE Object
        DESCRIPTION. A Beam object to be plotted
    optical_system : TYPE list
        DESCRIPTION. A list of SphericalRefraction objects which comprise the
                     optical system to be analysed.
                     *Note, should not include an output plane.
    z_0 : TYPE float (optional)
        DESCRIPTION. Arbitrary location of the distant output plane. 
                     The default is 1.0. If the focus of the optical system is 
                     likely to be greater than 1m then increase this quantity.
                     
    test_ray_dev : TYPE float (optional)
        DESCRIPTION. The deviation of the test ray from the z-axis. The test 
                     ray should be arbitrary close to the z-axis without being 
                     along it. The default is 1e-6.
    Returns
    -------
    rms_rad : TYPE float
        DESCRIPTION. The RMS beam radius at the beam focus

    '''
    lab1 = 'rays'
    if label is not None:
        lab1 = label + ' Rays'
    # the get_rms_size and focal_point methods take care of all the typechecks
    rms_size = get_rms_size(beam, optical_system, z_0, test_ray_dev)
    plt.scatter(beam.vertices()[:,-1,0], beam.vertices()[:,-1,1], label=lab1)
    thetas = np.linspace(0, 2*np.pi)
    all_x = rms_size * np.sin(thetas)
    all_y = rms_size * np.cos(thetas)
    plt.plot(all_x, all_y, color='black')
    plt.legend()
    return rms_size

def plano_convex_optimise(lens_params, init_curv_guesses, beam_params, z_0=1.0, 
                          test_ray_dev=1e-6):
    '''
    An optimisation function, which calculates the desired curvatures of a
    singlet lens by minimising the RMS radius at the focal point. It does so 
    for a given specified beam object. Note that the lens_params input is a 
    list that requires the following form,
    
    [z_0, thickness, external_ref_index, internal_ref_index, aperture_radius]
    
    where z_0 is the point where the surface intersects with the z axis. The 
    beam_params input requires the fomat,
    
    [rays_per_layer, radius, num_layers]
    
    Also, init_curv_guesses is a list with format 
    
    [surface_1_curvature, surface_2_curvature]

    Parameters
    ----------
    lens_params : TYPE list
        DESCRIPTION. A list containing all relevant information regarding the
                     singlet configuration (see above for format)
    init_curv_guesses : TYPE
        DESCRIPTION. A list containing initial curvature guesses (see above for 
                     format)
    beam_params : TYPE list
        DESCRIPTION. A list containing all relevant information regarding the
                     Beam for which the lens is optimised for (see above for 
                     format).
    z_0 : TYPE float (optional)
        DESCRIPTION. Arbitrary location of the distant output plane. 
                     The default is 1.0. If the focus of the optical system is 
                     likely to be greater than 1m then increase this quantity.
                     
    test_ray_dev : TYPE float (optional)
        DESCRIPTION. The deviation of the test ray from the z-axis. The test 
                     ray should be arbitrary close to the z-axis without being 
                     along it. The default is 1e-6.
    Returns
    -------
    TYPE OptimizeResult Object
        DESCRIPTION. Returns an OptimizeResult object which contains all
                     relevent information regarding the optimisation. From 
                     scipy documentation, 
                     "Important attributes are: x the 
                     solution array, success a Boolean flag indicating if the 
                     optimizer exited successfully and message which describes 
                     the cause of the termination. See OptimizeResult for a 
                     description of other attributes."
                     Best to print the outcome to make sense of result. 
    '''
    if not isinstance(lens_params, list):
        raise TypeError('lens_params_input must be a list of length 4 with '
                        + 'format [z_0, thickness, external_ref_index, '
                        + 'internal_ref_index, aperture_radius]')
    elif len(lens_params) != 5:
        raise ValueError('lens_params_input must be a list of length 5 with '
                        + 'format [z_0, thickness, external_ref_index, '
                        + 'internal_ref_index, aperture_radius]')
    if not isinstance(init_curv_guesses, list):
        raise TypeError('init_curv_guesses must be a list of length 2 with '
                        + 'format [surface_1_curvature, surface_2_curvature]')
    elif len(init_curv_guesses) != 2:
        raise ValueError('init_curv_guesses must be a list of length 2 with '
                        + 'format [surface_1_curvature, surface_2_curvature]')
    if not isinstance(beam_params, list):
        raise TypeError('beam_params must be a list of length 3 with '
                        + 'format [rays_per_layer, radius, num_layers]')
    elif len(beam_params) != 3:
        raise ValueError('init_curv_guesses must be a list of length 2 with '
                        + 'format [rays_per_layer, radius, num_layers]')
    # z_0 and test_ray_dev are typechecked by the get_rms_size method.
    init_surface_pos = lens_params[0]
    final_surface_pos = lens_params[0] + lens_params[1]
    external_ref_index = lens_params[2]
    internal_ref_index = lens_params[3]
    aperture_rad = lens_params[4]
    rays_per_layer = beam_params[0]
    beam_rad = beam_params[1]
    no_layers = beam_params[2]
    
    def rms_size(curvatures):
        '''
        The function for RMS size which is input into the 
        scipy.optimize.minimize() function i.e. the RMS size is minimized.

        Parameters
        ----------
        curvatures : TYPE list
            DESCRIPTION. A list of length 2 which contains curvatures for each
                         lens surface.
        Returns
        -------
        rms_radius : TYPE float
            DESCRIPTION. The RMS radius calculated for a given surface 
                         curvature combination

        '''
        plano_convex_lens = [SphericalRefraction(init_surface_pos, 
                                                 curvatures[0], 
                                                 external_ref_index, 
                                                 internal_ref_index, 
                                                 aperture_rad),
                             SphericalRefraction(final_surface_pos, 
                                                 curvatures[1], 
                                                 internal_ref_index,
                                                 external_ref_index, 
                                                 aperture_rad)]
        rms_radius = get_rms_size(Beam(rays_per_layer, beam_rad, no_layers), 
                                  plano_convex_lens)
        return rms_radius
    
    result = fmin(rms_size, init_curv_guesses)
    return result

