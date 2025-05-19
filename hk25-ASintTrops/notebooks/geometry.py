import numpy as np
from astropy.convolution import convolve, Gaussian2DKernel

def grad_sphere(field, llon, llat):
    """
    Function to calculate the gradient of a 2D scalar field over a sphere, given the coordinates in degrees on 
    the same 2D grid. The derivatives are taken as second-order differences in the interior, and first-order 
    (forward or backward) on the edges.
    """
    R = 6371.0e3 # Earth radius in km.
    
    field = np.double(field)
    llon = np.double(llon)
    llat = np.double(llat)
    
    costheta = np.cos(llat*np.pi/180)
    
    df_dx = field-field
    df_dx[:,1:-1] = (field[:,2:]-field[:,:-2])/(R*costheta[:,1:-1]*(llon[:,2:]-llon[:,:-2])*np.pi/180)
    df_dx[:,0] = (field[:,1]-field[:,0])/(R*costheta[:,0]*(llon[:,1]-llon[:,0])*np.pi/180)
    df_dx[:,-1] = (field[:,-1]-field[:,-2])/(R*costheta[:,-1]*(llon[:,-1]-llon[:,-2])*np.pi/180)
    
    df_dy = field-field
    df_dy[1:-1,:] = (field[2:,:]-field[:-2,:])/(R*(llat[2:,:]-llat[:-2,:])*np.pi/180)
    df_dy[0,:] = (field[1,:]-field[0,:])/(R*(llat[1,:]-llat[0,:])*np.pi/180)
    df_dy[-1,:] = (field[-1,:]-field[-2,:])/(R*(llat[-1,:]-llat[-2,:])*np.pi/180)
    
    return df_dx, df_dy

def div_sphere(field_a, field_b, llon, llat):
    """
    Function to calculate the divergence of a 2D vectorial field over a sphere, given the coordinates in degrees on 
    the same 2D grid. The derivatives are taken as second-order differences in the interior, and first-order 
    (forward or backward) on the edges.
    """
    R = 6371.0e3 # Earth radius in km.
    
    field_a = np.double(field_a)
    field_b = np.double(field_b)
    llon = np.double(llon)
    llat = np.double(llat)
    
    costheta = np.cos(llat*np.pi/180)

    div_a = field_a-field_a
    div_a[:,1:-1] = (field_a[:,2:]-field_a[:,:-2])/(R*costheta[:,1:-1]*(llon[:,2:]-llon[:,:-2])*np.pi/180)
    div_a[:,0] = (field_a[:,1]-field_a[:,0])/(R*costheta[:,0]*(llon[:,1]-llon[:,0])*np.pi/180)
    div_a[:,-1] = (field_a[:,-1]-field_a[:,-2])/(R*costheta[:,-1]*(llon[:,-1]-llon[:,-2])*np.pi/180)
    
    div_b = field_b-field_b
    div_b[1:-1,:] = (field_b[2:,:]*costheta[2:,:]-field_b[:-2,:]*costheta[:-2,:])/(R*costheta[1:-1,:]*(llat[2:,:]-llat[:-2,:])*np.pi/180)
    div_b[0,:] = (field_b[1,:]*costheta[1,:]-field_b[0,:]*costheta[0,:])/(R*costheta[0,:]*(llat[1,:]-llat[0,:])*np.pi/180)
    div_b[-1,:] = (field_b[-1,:]*costheta[-1,:]-field_b[-2,:]*costheta[-2,:])/(R*costheta[-1,:]*(llat[-1,:]-llat[-2,:])*np.pi/180)
        
    div = div_a + div_b
    return div
    
def stretched_div_sphere(field_a, field_b, llon, llat, u_field, v_field):
    """
    Function to calculate the divergence of a 2D vectorial field over a sphere, given the coordinates in degrees on 
    the same 2D grid. The coordinates are stretched with the u_field and the v_field components.
    The derivatives are taken as second-order differences in the interior, and first-order 
    (forward or backward) on the edges.
    """
    R = 6371.0e3 # Earth radius in km.

    field_a = np.double(field_a)
    field_b = np.double(field_b)
    llon = np.double(llon)
    llat = np.double(llat)
    u_field = np.double(u_field)
    v_field = np.double(v_field)
    
    costheta = np.cos(llat*np.pi/180)

    div_a = field_a-field_a
    div_a[:,1:-1] = (field_a[:,2:]-field_a[:,:-2])/(R*costheta[:,1:-1]*(llon[:,2:]-llon[:,:-2])*np.pi/180)
    div_a[:,0] = (field_a[:,1]-field_a[:,0])/(R*costheta[:,0]*(llon[:,1]-llon[:,0])*np.pi/180)
    div_a[:,-1] = (field_a[:,-1]-field_a[:,-2])/(R*costheta[:,-1]*(llon[:,-1]-llon[:,-2])*np.pi/180)
    
    div_b = field_b-field_b
    div_b[1:-1,:] = (field_b[2:,:]*costheta[2:,:]-field_b[:-2,:]*costheta[:-2,:])/(R*costheta[1:-1,:]*(llat[2:,:]-llat[:-2,:])*np.pi/180)
    div_b[0,:] = (field_b[1,:]*costheta[1,:]-field_b[0,:]*costheta[0,:])/(R*costheta[0,:]*(llat[1,:]-llat[0,:])*np.pi/180)
    div_b[-1,:] = (field_b[-1,:]*costheta[-1,:]-field_b[-2,:]*costheta[-2,:])/(R*costheta[-1,:]*(llat[-1,:]-llat[-2,:])*np.pi/180)
        
    div = np.abs(u_field)*div_a + np.abs(v_field)*div_b
    return div

def nan_gaussian_filter(field,sigma):
    """
    Function to smooth the field ignoring the NaNs.
    - For sigma>10 I follow the first answer here: 
    https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    By default, the filter is truncated at 4 sigmas.
    Lower computational cost, at the cost of a worse nans management.
    - For sigma<10 I use astropy, which has a better nans management, 
    but the bad habit of extending the fields beyond the original 
    boundaries (quickfixed).
    """
    if sigma >10:
        from scipy.ndimage import gaussian_filter
        
        field = np.double(field)
        
        # Take the original field and replace the NaNs with zeros.
        field0 = field.copy()
        field0[np.isnan(field)] = 0
        ff = gaussian_filter(field0, sigma=sigma)
        
        # Create the smoothed weight field.
        weight = 0*field.copy()+1   # here you have either nan or 1
        weight[np.isnan(field)] = 0 # here you have only 0 and 1
        ww = gaussian_filter(weight, sigma=sigma)   # smoothed weight field
        # The product ww*weight has 0 where you had nans, and a smoothed ]0,1] value in the other positions.
        # Dividing by ww*weight you retrieve the original nans (div by 0 == nan) and 
        # rescale up the values which where resized by fake zeros in ff.  
        zz = ff/(ww*weight) # This rescale for the actual weights used in the filter and set to NaN where the field
                            # was originally NaN.
        return zz
    else: 
        # Define the Gaussian kernel with given sigma
        kernel = Gaussian2DKernel(x_stddev=sigma)
        # Apply convolution
        convolved_field = convolve(field, kernel, boundary='fill',fill_value = np.nan)
        # Remove extensions (even with boundary='fill' there are some.
        convolved_field[np.isnan(field)] = np.nan
        return convolved_field

