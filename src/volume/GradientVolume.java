/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package volume;

import com.jogamp.opengl.math.VectorUtil;
import util.VectorMath;

/**
 *
 * @author michel and modified by Anna Vilanova 
 * 
 * 
 */

//////////////////////////////////////////////////////////////////////
///////////////// CONTAINS FUNCTIONS TO BE IMPLEMENTED ///////////////
//////////////////////////////////////////////////////////////////////
public class GradientVolume {

	
    
//Do NOT modify this attributes
    private int dimX, dimY, dimZ;
    private VoxelGradient zero = new VoxelGradient();
    VoxelGradient[] data;
    Volume volume;
    double maxmag;
    
//If needed add new attributes here:


    //Do NOT modify this function
    // 
    // Computes the gradient of the volume attribute and save it into the data attribute
    // This is a lengthy computation and is performed only once (have a look at the constructor GradientVolume) 
    //
    private void compute() {

        for (int i=0; i<data.length; i++) {
            data[i] = zero;
        }
       
        for (int z=1; z<dimZ-1; z++) {
            for (int y=1; y<dimY-1; y++) {
                for (int x=1; x<dimX-1; x++) {
                    float gx = (volume.getVoxel(x+1, y, z) - volume.getVoxel(x-1, y, z))/2.0f;
                    float gy = (volume.getVoxel(x, y+1, z) - volume.getVoxel(x, y-1, z))/2.0f;
                    float gz = (volume.getVoxel(x, y, z+1) - volume.getVoxel(x, y, z-1))/2.0f;
                    VoxelGradient grad = new VoxelGradient(gx, gy, gz);
                    setGradient(x, y, z, grad);
                }
            }
        }
        maxmag=calculateMaxGradientMagnitude();
     }
    	
    
//////////////////////////////////////////////////////////////////////
///////////////// FUNCTION TO BE IMPLEMENTED /////////////////////////
//////////////////////////////////////////////////////////////////////
//This function linearly interpolates gradient vector g0 and g1 given the factor (t) 
//the resut is given at result. You can use it to tri-linearly interpolate the gradient 
    
	private void interpolate(VoxelGradient g0, VoxelGradient g1, float factor, VoxelGradient result) {
            
        result.x = g1.x*factor+g0.x*(1-factor);
        result.y = g1.y*factor+g0.y*(1-factor);
        result.z = g1.z*factor+g0.z*(1-factor);
        result.mag = (float) Math.sqrt(result.x * result.x + result.y * result.y + result.z * result.z);
    }
	
//////////////////////////////////////////////////////////////////////
///////////////// FUNCTION TO BE IMPLEMENTED /////////////////////////
//////////////////////////////////////////////////////////////////////
// This function should return linearly interpolated gradient for position coord[]
// right now it returns the nearest neighbour        
        
    public VoxelGradient getGradient(double[] coord) {
        //make sure not on edges
        if (coord[0] < 0 || coord[0] > (dimX-2) || coord[1] < 0 || coord[1] > (dimY-2)
                || coord[2] < 0 || coord[2] > (dimZ-2)) {
            return zero;
        }
        /* notice that in this framework we assume that the distance between neighbouring voxels is 1 in all directions*/
        int x = (int) Math.floor(coord[0]);
        int y = (int) Math.floor(coord[1]);
        int z = (int) Math.floor(coord[2]);

        float fac_x = (float) coord[0] - x;
        float fac_y = (float) coord[1] - y;
        float fac_z = (float) coord[2] - z;

        VoxelGradient r0,r1,r2,r3,r4,r5,r6;
        r0=r1=r2=r3=r4=r5=r6= new VoxelGradient();

        interpolate(getGradient(x, y, z), getGradient(x+1, y, z), fac_x, r0);
        interpolate(getGradient(x, y+1, z), getGradient(x+1, y+1, z), fac_x, r1);
        interpolate(getGradient(x, y, z+1), getGradient(x+1, y, z+1), fac_x, r2);
        interpolate(getGradient(x, y+1, z+1), getGradient(x+1, y+1, z+1), fac_x, r3);
        interpolate(r0, r1, fac_y, r4);
        interpolate(r2, r3, fac_y, r5);
        interpolate(r4, r5, fac_z, r6);

        return r6;
    }
    
    
    
    //Do NOT modify this function
    public VoxelGradient getGradientNN(double[] coord) {
        if (coord[0] < 0 || coord[0] > (dimX-2) || coord[1] < 0 || coord[1] > (dimY-2)
                || coord[2] < 0 || coord[2] > (dimZ-2)) {
            return zero;
        }

        int x = (int) Math.round(coord[0]);
        int y = (int) Math.round(coord[1]);
        int z = (int) Math.round(coord[2]);
        return getGradient(x, y, z);
    }
    
    //Returns the maximum gradient magnitude
    //The data array contains all the gradients, in this function you have to return the maximum magnitude of the vectors in data[] 
    //Do NOT modify this function
    private double calculateMaxGradientMagnitude() {
        if (maxmag >= 0) {
            return maxmag;
        } else {
            double magnitude = data[0].mag;
            for (int i=0; i<data.length; i++) {
                magnitude = data[i].mag > magnitude ? data[i].mag : magnitude;
            }   
            maxmag = magnitude;
            return magnitude;
        }
    }
    
    //Do NOT modify this function
    public double getMaxGradientMagnitude()
    {
        return this.maxmag;
    }
	//////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////
	
	
	//Do NOT modify this function
	public GradientVolume(Volume vol) {
        volume = vol;
        dimX = vol.getDimX();
        dimY = vol.getDimY();
        dimZ = vol.getDimZ();
        data = new VoxelGradient[dimX * dimY * dimZ];
        maxmag = -1.0;
        compute();
    }

	//Do NOT modify this function
	public VoxelGradient getGradient(int x, int y, int z) {
        return data[x + dimX * (y + dimY * z)];
    }

  
  
    //Do NOT modify this function
    public void setGradient(int x, int y, int z, VoxelGradient value) {
        data[x + dimX * (y + dimY * z)] = value;
    }

    //Do NOT modify this function
    public void setVoxel(int i, VoxelGradient value) {
        data[i] = value;
    }
    
    //Do NOT modify this function
    public VoxelGradient getVoxel(int i) {
        return data[i];
    }
    
    //Do NOT modify this function
    public int getDimX() {
        return dimX;
    }
    
    //Do NOT modify this function
    public int getDimY() {
        return dimY;
    }
    
    //Do NOT modify this function
    public int getDimZ() {
        return dimZ;
    }

}
